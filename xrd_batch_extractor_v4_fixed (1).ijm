// ============================================================================
// XRD Batch Extractor for Fiji — v4
// ============================================================================
// Workflow:
//   1. You calibrate axes (click 4 tick marks) — ONCE per graph type
//   2. Script auto-extracts curve from current image
//   3. Auto-exports CSV with 2theta + intensity
//   4. Auto-detects peaks and exports peak list
//   5. Batch mode: processes entire folder with same calibration
// ============================================================================

// ── GLOBALS ─────────────────────────────────────────────────────────────────
var px1, py1, px2, py2;          // pixel positions of calibration points
var xVal1, yVal1, xVal2, yVal2;  // real-world values of calibration points
var roiX, roiY, roiW, roiH;     // plot region rectangle
var threshold;                   // brightness threshold for curve detection
var peakWindow;                  // peak detection window (in data points)
var peakMinHeight;               // minimum peak prominence

// ============================================================================
// MAIN — Choose mode
// ============================================================================
macro "XRD Batch Extractor" {

    modes = newArray("Single image (calibrate + extract)",
                     "Batch folder (calibrate once, extract all)");
    Dialog.create("XRD Extractor v4");
    Dialog.addChoice("Mode:", modes, modes[0]);
    Dialog.addNumber("Brightness threshold (0-255):", 100);
    Dialog.addNumber("Peak detection window (data points):", 15);
    Dialog.addNumber("Peak min prominence (counts):", 50);
    Dialog.show();

    mode = Dialog.getChoice();
    threshold = Dialog.getNumber();
    peakWindow = Dialog.getNumber();
    peakMinHeight = Dialog.getNumber();

    if (mode == modes[0]) {
        // ── Single image mode ──
        if (nImages == 0) {
            exit("Open an XRD graph image first, then run this macro.");
        }
        calibrateAxes();
        csvPath = extractAndSave("");
        if (csvPath != "") {
            showMessage("Done!", "Extracted to:\n" + csvPath);
        }
    } else {
        // ── Batch mode ──
        inputDir = getDirectory("Select FOLDER with XRD images");
        outputDir = getDirectory("Select OUTPUT folder for CSVs");

        // Calibrate on first image
        fileList = getFileList(inputDir);
        imageFiles = filterImageFiles(fileList);
        if (imageFiles.length == 0) {
            exit("No image files found in folder.");
        }

        // Open first image for calibration
        open(inputDir + imageFiles[0]);
        showMessage("Calibration", "Calibrate axes on this first image.\n" +
                     "The same calibration will be used for ALL images in the folder.");
        calibrateAxes();

        // Extract first image
        extractAndSave(outputDir);
        close();

        // Process remaining images with same calibration
        for (i = 1; i < imageFiles.length; i++) {
            showProgress(i, imageFiles.length);
            showStatus("Processing " + (i+1) + "/" + imageFiles.length + ": " + imageFiles[i]);
            open(inputDir + imageFiles[i]);
            extractAndSave(outputDir);
            close();
        }

        showMessage("Batch Complete!",
            imageFiles.length + " images processed.\nCSVs saved to:\n" + outputDir);
    }
}

// ============================================================================
// CALIBRATE AXES — Click on 4 tick marks
// ============================================================================
function calibrateAxes() {
    // Step 1: Draw rectangle around plot area
    setTool("rectangle");
    waitForUser("Step 1: Draw ROI",
        "Draw a rectangle around the PLOT AREA only.\n" +
        "(Exclude axis labels and title)\n\n" +
        "Then click OK.");

    getSelectionBounds(roiX, roiY, roiW, roiH);
    if (roiW == 0 || roiH == 0) {
        exit("No rectangle selected. Please draw a rectangle and try again.");
    }

    // Step 2: Click X-axis calibration points
    setTool("point");
    waitForUser("Step 2: X-axis Point 1",
        "Click on a KNOWN TICK MARK on the X-axis.\n" +
        "(e.g., click on the '5' tick mark)\n\n" +
        "Then click OK.");
    getSelectionCoordinates(xpoints, ypoints);
    px1 = xpoints[0];

    waitForUser("Step 2: X-axis Point 2",
        "Click on ANOTHER TICK MARK on the X-axis.\n" +
        "(e.g., click on the '40' tick mark)\n\n" +
        "Then click OK.");
    getSelectionCoordinates(xpoints, ypoints);
    px2 = xpoints[0];

    // Step 3: Enter X values
    Dialog.create("X-axis Values");
    Dialog.addNumber("Value at Point 1 (left tick):", 5);
    Dialog.addNumber("Value at Point 2 (right tick):", 40);
    Dialog.show();
    xVal1 = Dialog.getNumber();
    xVal2 = Dialog.getNumber();

    // Step 4: Click Y-axis calibration points
    setTool("point");
    waitForUser("Step 3: Y-axis Point 1",
        "Click on a KNOWN TICK MARK on the Y-axis.\n" +
        "(e.g., click on the '0' tick mark)\n\n" +
        "Then click OK.");
    getSelectionCoordinates(xpoints, ypoints);
    py1 = ypoints[0];

    waitForUser("Step 3: Y-axis Point 2",
        "Click on ANOTHER TICK MARK on the Y-axis.\n" +
        "(e.g., click on the '5000' tick mark)\n\n" +
        "Then click OK.");
    getSelectionCoordinates(xpoints, ypoints);
    py2 = ypoints[0];

    // Step 5: Enter Y values
    Dialog.create("Y-axis Values");
    Dialog.addNumber("Value at Point 1 (bottom tick):", 0);
    Dialog.addNumber("Value at Point 2 (top tick):", 5000);
    Dialog.show();
    yVal1 = Dialog.getNumber();
    yVal2 = Dialog.getNumber();

    run("Select None");

    print("=== Calibration Complete ===");
    print("  ROI: x=" + roiX + " y=" + roiY + " w=" + roiW + " h=" + roiH);
    print("  X: pixel " + px1 + "=" + xVal1 + ",  pixel " + px2 + "=" + xVal2);
    print("  Y: pixel " + py1 + "=" + yVal1 + ",  pixel " + py2 + "=" + yVal2);
}

// ============================================================================
// EXTRACT CURVE + EXPORT CSV + DETECT PEAKS
// ============================================================================
function extractAndSave(outputDir) {
    title = getTitle();
    baseName = replace(title, "\\.[^.]+$", "");  // strip extension

    // ── Build pixel-to-value mapping ──
    xScale = (xVal2 - xVal1) / (px2 - px1);
    xOffset = xVal1 - xScale * px1;
    yScale = (yVal2 - yVal1) / (py2 - py1);
    yOffset = yVal1 - yScale * py1;

    // ── Extract curve: topmost dark pixel per column ──
    nCols = roiW;
    xData = newArray(nCols);
    yData = newArray(nCols);
    validCount = 0;

    for (col = 0; col < nCols; col++) {
        imgX = roiX + col;
        // Scan from top of ROI downward — find topmost dark pixel
        foundY = -1;
        for (row = 0; row < roiH; row++) {
            imgY = roiY + row;
            pixVal = getPixel(imgX, imgY);

            // Handle RGB or grayscale
            if (bitDepth() == 24) {
                r = (pixVal >> 16) & 0xff;
                g = (pixVal >> 8) & 0xff;
                b = pixVal & 0xff;
                brightness = (r + g + b) / 3;
            } else {
                brightness = pixVal;
            }

            if (brightness < threshold) {
                foundY = imgY;
                row = roiH;  // exit loop (no break in IJM)
            }
        }

        if (foundY >= 0) {
            xReal = xScale * imgX + xOffset;
            yReal = yScale * foundY + yOffset;
            xData[validCount] = xReal;
            yData[validCount] = yReal;
            validCount++;
        }
    }

    if (validCount == 0) {
        print("WARNING: No curve detected in " + title + ". Try lowering threshold.");
        return "";
    }

    // ── Trim arrays to valid data ──
    xTrimmed = Array.trim(xData, validCount);
    yTrimmed = Array.trim(yData, validCount);

    // ── Determine output path ──
    if (outputDir == "") {
        // Single mode — ask user for output folder and filename
        saveDir = getDirectory("Choose folder to save CSV");
        Dialog.create("Save CSV");
        Dialog.addString("Filename:", baseName + "_extracted.csv");
        Dialog.show();
        csvName = Dialog.getString();
        csvPath = saveDir + csvName;
    } else {
        csvPath = outputDir + baseName + "_extracted.csv";
    }

    // ── Write data CSV ──
    f = File.open(csvPath);
    print(f, "2theta_deg,Intensity_counts");
    for (i = 0; i < validCount; i++) {
        print(f, d2s(xTrimmed[i], 4) + "," + d2s(yTrimmed[i], 1));
    }
    File.close(f);
    print("Saved: " + csvPath + "  (" + validCount + " points)");

    // ── Peak Detection ──
    peakPositions = newArray(0);
    peakIntensities = newArray(0);
    peakCount = 0;

    // Simple local-maximum peak finder
    halfWin = floor(peakWindow / 2);
    for (i = halfWin; i < validCount - halfWin; i++) {
        currentY = yTrimmed[i];
        isPeak = true;

        // Check if current point is maximum in window
        for (j = i - halfWin; j <= i + halfWin; j++) {
            if (j != i && yTrimmed[j] >= currentY) {
                isPeak = false;
                j = i + halfWin + 1;  // exit loop (no break in IJM)
            }
        }

        if (isPeak) {
            // Check prominence: compare to minimum in left and right windows
            leftMin = currentY;
            rightMin = currentY;
            for (j = i - halfWin; j < i; j++) {
                if (yTrimmed[j] < leftMin) leftMin = yTrimmed[j];
            }
            for (j = i + 1; j <= i + halfWin; j++) {
                if (yTrimmed[j] < rightMin) rightMin = yTrimmed[j];
            }
            prominence = currentY - maxOf(leftMin, rightMin);

            if (prominence >= peakMinHeight) {
                peakPositions = Array.concat(peakPositions, xTrimmed[i]);
                peakIntensities = Array.concat(peakIntensities, currentY);
                peakCount++;
            }
        }
    }

    // ── Write peaks CSV ──
    peakPath = replace(csvPath, "_extracted.csv", "_peaks.csv");
    f2 = File.open(peakPath);
    print(f2, "Peak_No,2theta_deg,Intensity_counts,Prominence_threshold");
    for (i = 0; i < peakCount; i++) {
        print(f2, (i+1) + "," + d2s(peakPositions[i], 3) + "," +
              d2s(peakIntensities[i], 1) + "," + d2s(peakMinHeight, 1));
    }
    File.close(f2);
    print("Peaks: " + peakCount + " found → " + peakPath);

    // ── Print peak summary to Log ──
    print("--- Peaks for " + title + " ---");
    for (i = 0; i < peakCount; i++) {
        print("  Peak " + (i+1) + ":  2θ = " + d2s(peakPositions[i], 3) +
              "°   I = " + d2s(peakIntensities[i], 0));
    }
    print("---");

    return csvPath;
}

// ============================================================================
// HELPER — Filter image files from directory listing
// ============================================================================
function filterImageFiles(fileList) {
    result = newArray(0);
    extensions = newArray(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif");
    for (i = 0; i < fileList.length; i++) {
        name = toLowerCase(fileList[i]);
        for (e = 0; e < extensions.length; e++) {
            if (endsWith(name, extensions[e])) {
                result = Array.concat(result, fileList[i]);
                e = extensions.length;  // exit loop (no break in IJM)
            }
        }
    }
    return result;
}

// ============================================================================
// HELPER — Max of two values
// ============================================================================
function maxOf(a, b) {
    if (a > b) return a;
    return b;
}
