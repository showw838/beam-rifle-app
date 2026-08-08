// State
let isRunning = false;
let videoStream = null;
let shotHistory = JSON.parse(localStorage.getItem('beamRifleHistory')) || [];
let scoreBuffer = []; // Debounce buffer for OCR

// WebRTC Sync State
let peer = null;
let conn = null;

function generatePIN() {
    return Math.floor(1000 + Math.random() * 9000).toString();
}

// Mode Selection UI Logic
document.querySelectorAll('input[name="appMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        const isCamera = e.target.value === 'camera';
        document.getElementById('cameraConnectionUI').style.display = isCamera ? 'block' : 'none';
        document.getElementById('monitorConnectionUI').style.display = !isCamera ? 'block' : 'none';
        
        const mainEl = document.getElementById('mainContent');
        if (isCamera) {
            document.querySelector('.camera-section').style.display = 'flex';
            mainEl.style.gridTemplateColumns = ''; // Reset to CSS default (responsive)
        } else {
            document.querySelector('.camera-section').style.display = 'none';
            // On PC, we want Target + History side by side when camera is hidden
            if (window.innerWidth > 1024) {
                mainEl.style.gridTemplateColumns = '1fr 300px';
            } else {
                mainEl.style.gridTemplateColumns = '';
            }
        }
        
        if (!isCamera && videoStream) {
            videoStream.getTracks().forEach(t => t.stop());
            videoStream = null;
            isRunning = false;
        }
    });
});

document.getElementById('hostBtn').addEventListener('click', () => {
    if (peer) peer.destroy();
    const pin = generatePIN();
    peer = new Peer('beam-rifle-' + pin);
    
    const display = document.getElementById('hostPinDisplay');
    display.innerText = `準備中... (PIN: ${pin})`;
    display.style.color = '#ffcc00';
    
    peer.on('open', () => {
        display.innerText = `接続待機中 (PIN: ${pin})`;
    });
    
    peer.on('connection', (connection) => {
        conn = connection;
        display.innerText = `接続完了！ (PIN: ${pin})`;
        display.style.color = '#00ff00';
    });
});

document.getElementById('joinBtn').addEventListener('click', () => {
    const pin = document.getElementById('joinPinInput').value.trim();
    if (pin.length !== 4) return alert("4桁のPINを入力してください。");
    
    const status = document.getElementById('joinStatus');
    status.innerText = "接続中...";
    status.style.color = '#ffcc00';
    
    if (peer) peer.destroy();
    peer = new Peer();
    
    peer.on('open', () => {
        conn = peer.connect('beam-rifle-' + pin);
        
        conn.on('open', () => {
            status.innerText = "接続完了！";
            status.style.color = '#00ff00';
        });
        
        conn.on('data', (data) => {
            if (data.type === 'score') {
                document.getElementById('currentScore').innerText = data.score.toFixed(1);
                recordShot(data.score, data.angle || 0);
            }
        });
        
        conn.on('error', (err) => {
             status.innerText = "エラー: " + err;
             status.style.color = '#ff0000';
        });
    });
});

// DOM Elements
const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const debugCanvas = document.getElementById('debugCanvas');
const startBtn = document.getElementById('startBtn');
const videoInput = document.getElementById('videoInput');
const currentScoreEl = document.getElementById('currentScore');
const historyList = document.getElementById('historyList');
const totalScoreEl = document.getElementById('totalScore');
const avgScoreEl = document.getElementById('avgScore');
const clearBtn = document.getElementById('clearBtn');
const vTarget = document.getElementById('virtualTarget');
const vCtx = vTarget.getContext('2d');


const toggleOcrBtn = document.getElementById('toggleOcrBtn');

let isOcrActive = false;

if (toggleOcrBtn) {
    toggleOcrBtn.addEventListener('click', () => {
        isOcrActive = !isOcrActive;
        if (isOcrActive) {
            toggleOcrBtn.innerText = '■ 読み取り停止';
            toggleOcrBtn.style.backgroundColor = '#ff2a2a';
            scoreBuffer = []; // Clear buffer when starting
        } else {
            toggleOcrBtn.innerText = '▶ 読み取り開始';
            toggleOcrBtn.style.backgroundColor = 'var(--primary-color)';
        }
    });
}


// Robust 7-segment parsing helper using Regional Density (Immune to slants!)
const checkRegion = (mat, x, y, width, height) => {
    let sx = Math.max(0, Math.min(Math.floor(x), mat.cols - 1));
    let sy = Math.max(0, Math.min(Math.floor(y), mat.rows - 1));
    let sw = Math.max(1, Math.min(Math.floor(width), mat.cols - sx));
    let sh = Math.max(1, Math.min(Math.floor(height), mat.rows - sy));
    
    let sample = mat.roi(new cv.Rect(sx, sy, sw, sh));
    let area = sw * sh;
    let whitePixels = cv.countNonZero(sample);
    sample.delete();
    // 0.15 is the perfect balance: low enough to catch dim segments in live video,
    // but high enough to ignore stray noise and slanted vertical bar crossovers.
    return whitePixels > (area * 0.15);
};

function read7SegmentRobust(roi, trueWidth) {
    let h = roi.rows;
    let w = trueWidth; 
    
    // Extremely narrow digits can only be '1' (e.g. aspect ratio < 0.35)
    if (w < h * 0.35) return "1";
    
    let segments = [false, false, false, false, false, false, false];
    
    // Dilate the isolated ROI to fill small gaps and thicken segments
    let dilatedRoi = new cv.Mat();
    let kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(roi, dilatedRoi, kernel);

    // Region definitions (width, height, x, y) designed to capture slanted segments
    // Narrow horizontal regions to strictly the center 30% so slanted vertical bars don't trigger them!
    // Expand vertical regions to 50% width to catch slanted lines.
    segments[0] = checkRegion(dilatedRoi, w * 0.35, 0, w * 0.3, h * 0.2); // Top
    segments[1] = checkRegion(dilatedRoi, 0, h * 0.2, w * 0.5, h * 0.25); // TL
    segments[2] = checkRegion(dilatedRoi, w * 0.5, h * 0.2, w * 0.5, h * 0.25); // TR
    segments[3] = checkRegion(dilatedRoi, w * 0.35, h * 0.4, w * 0.3, h * 0.2); // Mid
    segments[4] = checkRegion(dilatedRoi, 0, h * 0.55, w * 0.5, h * 0.25); // BL
    segments[5] = checkRegion(dilatedRoi, w * 0.5, h * 0.55, w * 0.5, h * 0.25); // BR
    segments[6] = checkRegion(dilatedRoi, w * 0.35, h * 0.8, w * 0.3, h * 0.2); // Bot

    dilatedRoi.delete();
    kernel.delete();
    
    let bitmask = 0;
    for (let i = 0; i < 7; i++) {
        if (segments[i]) bitmask |= (1 << i);
    }
    
    const digitMap = {
        // Standard digits
        0x77: "0", 0x24: "1", 0x5D: "2", 0x6D: "3",
        0x2E: "4", 0x6B: "5", 0x7B: "6", 0x25: "7",
        0x7F: "8", 0x6F: "9",
        
        // Error-correcting variations for live camera fluctuations (flickering segments)
        0x59: "2", // 2 missing TR
        0x4D: "2", // 2 missing BL
        0x5C: "2", // 2 missing Top
        
        0x75: "0", // 0 missing TL
        0x73: "0", // 0 missing TR
        0x67: "0", // 0 missing BL
        0x57: "0", // 0 missing BR
        0x76: "0", // 0 missing Top (U-shape)
        
        0x7A: "6", // 6 missing Top
        
        0x27: "7", // 7 with TL hook
        0x65: "7", // 7 with Bot (slanted leg hitting bottom center)
        
        0x2F: "9", // 9 missing Bot
        0x6E: "9"  // 9 missing Top
    };
    
    return digitMap[bitmask] || "?";
}
// Check if OpenCV loaded
setTimeout(() => {
    if (!window.ocvReady) {
        const errEl = document.getElementById('debugError');
        if (errEl) {
            errEl.style.display = 'block';
            errEl.innerText = "画像処理エンジン(OpenCV)の読み込みに失敗しました。ページをリロードしてください。";
        }
    }
}, 5000);

// Start Camera
startBtn.addEventListener('click', async () => {
    if (isRunning) return;
    
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } } 
        });
        video.srcObject = videoStream;
        document.querySelector('.video-container').classList.add('active');
        console.log('OpenCV is ready.');
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isRunning = true;
            processFrames();
        };
    } catch (err) {
        console.error("Camera error: ", err);
        alert("Camera access denied or not available.");
    }
});

// Load Test Video
videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const url = URL.createObjectURL(file);
    video.srcObject = null;
    video.src = url;
    video.loop = true;
    
    document.querySelector('.video-container').classList.add('active');
    console.log('OpenCV is ready.');
    
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        video.play().catch(e => console.error("Video play failed:", e));
        isRunning = true;
        processFrames();
    };
});

// Fixed Calibration Boxes (Base Ratios)
const scoreRect = { x: 0.1, y: 0.35, w: 0.35, h: 0.266 };
const targetRect = { x: 0.55, y: 0.35, w: 0.533, h: 0.533 }; // Square for the target

// Processing Loop
let lastProcessTime = 0;
let lastDetectedScore = null;
let lastDetectedAngle = 0;
const ctx = canvas.getContext('2d');

async function processFrames() {
    if (!isRunning) return;
    
    const now = Date.now();
    
    // --- 0. Zoom and Pan Video onto Canvas (EVERY FRAME) ---
    let zoom = parseFloat(document.getElementById('zoomSlider').value) || 1;
    let panX = parseFloat(document.getElementById('panXSlider').value) || 0;
    let panY = parseFloat(document.getElementById('panYSlider').value) || 0;

    let vw = video.videoWidth;
    let vh = video.videoHeight;

    if (vw > 0 && vh > 0) {
        let sw = vw / zoom;
        let sh = vh / zoom;
        let cx = vw / 2 + (panX * (vw - sw) / 2);
        let cy = vh / 2 + (panY * (vh - sh) / 2);
        let sx = cx - sw / 2;
        let sy = cy - sh / 2;

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    }

    // Use Math.min to ensure boxes don't become massive on extreme aspect ratios
    // and maintain the exact physical shape they had in the portrait video.
    let baseSize = Math.min(canvas.width, canvas.height);

    let blueX = parseFloat(document.getElementById('blueXSlider').value) || 0.1;
    let blueY = parseFloat(document.getElementById('blueYSlider').value) || 0.35;
    let blueScale = parseFloat(document.getElementById('blueScaleSlider').value) || 1;
    
    let pinkX = parseFloat(document.getElementById('pinkXSlider').value) || 0.55;
    let pinkY = parseFloat(document.getElementById('pinkYSlider').value) || 0.35;
    let pinkScale = parseFloat(document.getElementById('pinkScaleSlider').value) || 1;

    let sX = blueX * canvas.width;
    let sY = blueY * canvas.height;
    let sW = scoreRect.w * baseSize * blueScale;
    let sH = scoreRect.h * baseSize * blueScale;

    let tX = pinkX * canvas.width;
    let tY = pinkY * canvas.height;
    let tW = targetRect.w * baseSize * pinkScale;
    let tH = targetRect.h * baseSize * pinkScale;

    // Draw Static Overlays on canvas (Dashed lines)
    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 3;
    ctx.strokeRect(sX, sY, sW, sH);
    ctx.fillStyle = '#00ffff';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText("SCORE AREA", sX, sY - 10);

    ctx.strokeStyle = '#ff00ff';
    ctx.strokeRect(tX, tY, tW, tH);
    ctx.fillStyle = '#ff00ff';
    ctx.fillText("TARGET AREA", tX, tY - 10);
    ctx.setLineDash([]); // Reset dash for other drawings

    // --- STOP HERE IF AI IS NOT READY ---
    if (!window.ocvReady) {
        requestAnimationFrame(processFrames);
        return;
    }

    // Process 2 frames per second to save resources
    if (now - lastProcessTime > 500) {
        lastProcessTime = now;
        
        try {
            document.getElementById('debugError').style.display = 'none';

        let src = cv.imread(canvas);
        
        // Ensure rects are valid before cropping
        if (targetRect.w > 0 && targetRect.h > 0) {
            let safeTX = Math.max(0, Math.min(Math.floor(tX), src.cols - 1));
            let safeTY = Math.max(0, Math.min(Math.floor(tY), src.rows - 1));
            let safeTW = Math.min(Math.floor(tW), src.cols - safeTX);
            let safeTH = Math.min(Math.floor(tH), src.rows - safeTY);
            
            let targetRoiRect = new cv.Rect(safeTX, safeTY, safeTW, safeTH);
            // Ensure width and height are valid after bounds checks
            if (targetRoiRect.width > 0 && targetRoiRect.height > 0) {
                // --- 1. Robust Target Center & Red Dot Detection ---
                let rightHalf = src.roi(targetRoiRect);

                // --- 1-A. Target Center is the center of the pink box ---
                // Since the user can now manually adjust the pink box perfectly,
                // we use the exact center of the box to avoid logos throwing off the math.
                let targetCenter = { x: targetRoiRect.width / 2, y: targetRoiRect.height / 2 };
                
                // Draw center crosshair on canvas to help user align the pink box
                ctx.strokeStyle = '#ffff00';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                // Horizontal line
                ctx.moveTo(targetCenter.x + targetRoiRect.x - 15, targetCenter.y + targetRoiRect.y);
                ctx.lineTo(targetCenter.x + targetRoiRect.x + 15, targetCenter.y + targetRoiRect.y);
                // Vertical line
                ctx.moveTo(targetCenter.x + targetRoiRect.x, targetCenter.y + targetRoiRect.y - 15);
                ctx.lineTo(targetCenter.x + targetRoiRect.x, targetCenter.y + targetRoiRect.y + 15);
                ctx.stroke();
                
                // Center circle
                ctx.beginPath();
                ctx.arc(targetCenter.x + targetRoiRect.x, targetCenter.y + targetRoiRect.y, 4, 0, 2 * Math.PI);
                ctx.stroke();

                // --- 1-B. Find Red Dot (HSV Thresholding) ---
                let hsv = new cv.Mat();
                cv.cvtColor(rightHalf, hsv, cv.COLOR_RGBA2RGB);
                cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

                const threshVal = parseInt(document.getElementById('redThreshold').value) || 100;
                let mask = new cv.Mat();

                let low1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, threshVal, threshVal, 0]);
                let high1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [15, 255, 255, 0]);
                let mask1 = new cv.Mat();
                cv.inRange(hsv, low1, high1, mask1);

                let low2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [165, threshVal, threshVal, 0]);
                let high2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [180, 255, 255, 0]);
                let mask2 = new cv.Mat();
                cv.inRange(hsv, low2, high2, mask2);

                cv.bitwise_or(mask1, mask2, mask);

                hsv.delete(); low1.delete(); high1.delete(); mask1.delete();
                low2.delete(); high2.delete(); mask2.delete();

                let contours = new cv.MatVector();
                let hierarchy = new cv.Mat();
                cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

                let foundDot = false;
                let dotCenter = { x: 0, y: 0 };

                for (let i = 0; i < contours.size(); i++) {
                    let cnt = contours.get(i);
                    let area = cv.contourArea(cnt);
                    // Dot can be tiny
                    if (area > 2 && area < 1500) {
                        let rect = cv.boundingRect(cnt);
                        dotCenter.x = rect.x + rect.width / 2;
                        dotCenter.y = rect.y + rect.height / 2;
                        foundDot = true;

                        ctx.strokeStyle = '#00ff88';
                        ctx.lineWidth = 3;
                        ctx.beginPath();
                        ctx.arc(dotCenter.x + targetRoiRect.x, dotCenter.y + targetRoiRect.y, 10, 0, 2 * Math.PI);
                        ctx.stroke();
                        
                        // Draw line from center to dot
                        ctx.strokeStyle = '#ff00ff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(targetCenter.x + targetRoiRect.x, targetCenter.y + targetRoiRect.y);
                        ctx.lineTo(dotCenter.x + targetRoiRect.x, dotCenter.y + targetRoiRect.y);
                        ctx.stroke();
                        break;
                    }
                }
                
                if (foundDot) {
                    const dx = dotCenter.x - targetCenter.x;
                    const dy = dotCenter.y - targetCenter.y;
                    lastDetectedAngle = Math.atan2(dy, dx);
                }

                // Cleanup
                rightHalf.delete(); contours.delete(); hierarchy.delete(); mask.delete();
            }
        }
        
        // --- 2. OCR (Tesseract) ---
        if (scoreRect.w > 0 && scoreRect.h > 0) {
            let safeSX = Math.max(0, Math.min(Math.floor(sX), src.cols - 1));
            let safeSY = Math.max(0, Math.min(Math.floor(sY), src.rows - 1));
            let safeSW = Math.min(Math.floor(sW), src.cols - safeSX);
            let safeSH = Math.min(Math.floor(sH), src.rows - safeSY);
            
            let scoreRoiRect = new cv.Rect(safeSX, safeSY, safeSW, safeSH);
            if (scoreRoiRect.width > 0 && scoreRoiRect.height > 0) {
                let leftHalf = src.roi(scoreRoiRect);
                const ocrThresh = parseInt(document.getElementById('ocrThreshold').value, 10);
                
                // Use Red channel instead of grayscale
                let channelsL = new cv.MatVector();
                cv.split(leftHalf, channelsL);
                let grayL = channelsL.get(0); 
                
                let maskL = new cv.Mat();
                // Use THRESH_BINARY to make the bright red LED text WHITE and dark panel BLACK.
                cv.threshold(grayL, maskL, ocrThresh, 255, cv.THRESH_BINARY);
                
                // Show what the AI sees for debugging unconditionally
                cv.imshow(document.getElementById('debugCanvas'), maskL);
                
                if (isOcrActive) {
                    let contours = new cv.MatVector();
                    let hierarchy = new cv.Mat();
                    cv.findContours(maskL, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
                    
                    // First pass: find max contour height
                    let maxContourHeight = 1;
                    let boxes = [];
                    for (let i = 0; i < contours.size(); ++i) {
                        let rect = cv.boundingRect(contours.get(i));
                        if (rect.height > maxContourHeight) maxContourHeight = rect.height;
                        boxes.push({rect: rect, contourIndex: i});
                    }
                    
                    // Filter dust and extract perfectly separated digits using Top-Half Vertical Projection
                    let digitBoxes = [];
                    // We must create a reusable mask to isolate contours so adjacent slanted digits don't overlap
                    let isolatedMask = new cv.Mat.zeros(maskL.rows, maskL.cols, cv.CV_8UC1);
                    
                    for (let i = 0; i < boxes.length; ++i) {
                        let rect = boxes[i].rect;
                        if (rect.height >= maxContourHeight * 0.5) {
                            
                            // Isolate this specific contour to prevent slanted neighbors from bleeding into the bounding box
                            isolatedMask.setTo(new cv.Scalar(0));
                            cv.drawContours(isolatedMask, contours, boxes[i].contourIndex, new cv.Scalar(255), cv.FILLED);
                            // AND with the original mask to restore holes (since FILLED destroys holes like the middle of '0')
                            cv.bitwise_and(maskL, isolatedMask, isolatedMask);
                            
                            let centerY = rect.y + Math.floor(rect.height / 2);
                            let colSums = new Int32Array(rect.width);
                            
                            // Sum white pixels in the TOP HALF of the isolated contour
                            // This completely ignores the decimal point at the bottom!
                            for (let cx = 0; cx < rect.width; cx++) {
                                let sum = 0;
                                for (let cy = rect.y; cy < centerY; cy++) {
                                    if (isolatedMask.ucharPtr(cy, rect.x + cx)[0] > 128) sum++;
                                }
                                colSums[cx] = sum;
                            }
                            
                            // Find contiguous blocks of columns that have white pixels
                            let rawBlocks = [];
                            let inBlock = false;
                            let blockStartX = 0;
                            
                            for (let cx = 0; cx < rect.width; cx++) {
                                if (colSums[cx] > 0) {
                                    if (!inBlock) {
                                        inBlock = true;
                                        blockStartX = cx;
                                    }
                                } else {
                                    if (inBlock) {
                                        inBlock = false;
                                        rawBlocks.push({start: blockStartX, end: cx});
                                    }
                                }
                            }
                            if (inBlock) {
                                rawBlocks.push({start: blockStartX, end: rect.width});
                            }
                            
                            // Merge blocks separated by a small gap (e.g., <= 3 pixels) to handle broken segments
                            if (rawBlocks.length > 0) {
                                let mergedBlocks = [rawBlocks[0]];
                                for (let j = 1; j < rawBlocks.length; j++) {
                                    let last = mergedBlocks[mergedBlocks.length - 1];
                                    let curr = rawBlocks[j];
                                    if (curr.start - last.end <= 4) {
                                        last.end = curr.end; // merge
                                    } else {
                                        mergedBlocks.push(curr);
                                    }
                                }
                                
                                for (let b of mergedBlocks) {
                                    let blockW = b.end - b.start;
                                    if (blockW > 2) {
                                        digitBoxes.push({
                                            x: rect.x + b.start,
                                            y: rect.y,
                                            w: blockW,
                                            h: rect.height,
                                            contourIndex: boxes[i].contourIndex
                                        });
                                    }
                                }
                            }
                        }
                    }
                    
                    // Sort digits left to right
                    digitBoxes.sort((a, b) => a.x - b.x);
                    
                    let parsedDigits = "";
                    for (let b of digitBoxes) {
                        // Isolate the contour again for the final ROI parsing
                        isolatedMask.setTo(new cv.Scalar(0));
                        cv.drawContours(isolatedMask, contours, b.contourIndex, new cv.Scalar(255), cv.FILLED);
                        cv.bitwise_and(maskL, isolatedMask, isolatedMask);
                        
                        // roi perfectly bounds the pure digit without adjacent interference
                        let rect = new cv.Rect(b.x, b.y, b.w, b.h);
                        let roi = isolatedMask.roi(rect);
                        let digitStr = read7SegmentRobust(roi, b.w);
                        if (digitStr !== "?") {
                            parsedDigits += digitStr;
                        }
                        roi.delete();
                    }
                    
                    isolatedMask.delete();
                    
                    contours.delete();
                    hierarchy.delete();
                    
                    const rawEl = document.getElementById('rawOcrText');
                    if (parsedDigits.length >= 2) {
                        // Insert decimal point before the last digit (e.g. 102 -> 10.2)
                        let integerPart = parsedDigits.substring(0, parsedDigits.length - 1);
                        let fractionalPart = parsedDigits.substring(parsedDigits.length - 1);
                        let finalScoreStr = integerPart + "." + fractionalPart;
                        let numericValue = parseFloat(finalScoreStr);
                        
                        // Process numeric value
                        let validScore = null;
                        if (!isNaN(numericValue) && numericValue > 0 && numericValue <= 10.9) {
                            validScore = numericValue;
                        }
                        
                        scoreBuffer.push(validScore);
                        if (rawEl) rawEl.innerText = `Raw Score: ${finalScoreStr} (From ${parsedDigits})`;
                    } else {
                        if (rawEl) rawEl.innerText = `Raw Score: ? (Not enough digits: ${parsedDigits})`;
                        scoreBuffer.push(null);
                    }
                    
                    // Always shift buffer to maintain length of 5
                    if (scoreBuffer.length > 5) scoreBuffer.shift();

                    if (scoreBuffer.length === 5) {
                        const counts = {};
                        let maxCount = 0;
                        let mode = null;
                        for (let val of scoreBuffer) {
                            if (val === null) continue;
                            counts[val] = (counts[val] || 0) + 1;
                            if (counts[val] > maxCount) {
                                maxCount = counts[val];
                                mode = val;
                            }
                        }

                        // If screen is blank (maxCount === 0), reset lastDetectedScore so next shot can be the same number
                        if (maxCount === 0) {
                            lastDetectedScore = null;
                        } else if (maxCount >= 3 && mode !== lastDetectedScore) {
                            lastDetectedScore = mode;
                            document.getElementById('currentScore').innerText = mode.toFixed(1);
                            
                            // Send angle along with the score
                            let detectedAngle = lastDetectedAngle;
                            recordShot(mode, detectedAngle); 
                            
                            // Send score via WebRTC if connected as Camera Host
                            const appMode = document.querySelector('input[name="appMode"]:checked').value;
                            if (appMode === 'camera' && conn && conn.open) {
                                conn.send({ type: 'score', score: mode, angle: detectedAngle });
                            }
                        }
                    }
                } else {
                    const rawEl = document.getElementById('rawOcrText');
                    if (rawEl) rawEl.innerText = "Raw Score: (Paused - 読み取り待機中)";
                }
                
                // Cleanup current frame resources
                leftHalf.delete(); channelsL.delete(); grayL.delete(); maskL.delete();
            }
        }
        
        // Cleanup memory
        if (typeof src !== 'undefined' && src !== null) {
            src.delete();
        }

        } catch (err) {
            console.error("Frame Processing Error:", err);
            const errDiv = document.getElementById('debugError');
            errDiv.style.display = 'block';
            errDiv.innerText = "Error: " + (err.stack || err.toString() || "Unknown OpenCV error");
            // Do not throw, allow requestAnimationFrame to continue
        }
    }
    
    requestAnimationFrame(processFrames);
}

// Data Logic
function getRadiusFromScore(score) {
    if (score >= 10.0) {
        // 10.0 to 10.9: 1.0mm diameter (0.5mm radius) divided into 10 parts
        return 0.5 * (10.9 - score) / 0.9;
    } else if (score >= 1.0) {
        // 1.0 to 9.9
        let I = Math.floor(score);
        let D = score - I;
        let R_out = 0.5 + (10 - I) * 2.5; // Outer radius of ring I
        return R_out - D * (2.5 / 0.9);
    }
    return 0; // Miss or out of bounds
}

function recordShot(score, angle) {
    const shot = {
        id: Date.now(),
        score: score,
        angle: angle,
        timestamp: new Date().toISOString()
    };

    shotHistory.push(shot);
    localStorage.setItem('beamRifleHistory', JSON.stringify(shotHistory));

    updateUI();
}

function updateUI() {
    // List
    historyList.innerHTML = '';
    let sum = 0;
    
    // Sort descending by time
    const sorted = [...shotHistory].reverse();
    
    sorted.forEach((shot, index) => {
        sum += shot.score;
        const li = document.createElement('li');
        li.className = 'history-item';
        li.innerHTML = `
            <span class="shot-num">#${shotHistory.length - index}</span>
            <span class="shot-score">${shot.score.toFixed(1)}</span>
        `;
        historyList.appendChild(li);
    });
    
    // Stats
    totalScoreEl.innerText = sum.toFixed(1);
    if (shotHistory.length > 0) {
        avgScoreEl.innerText = (sum / shotHistory.length).toFixed(1);
    } else {
        avgScoreEl.innerText = "0.0";
    }
    
    drawVirtualTarget();
}

// Drawing the Virtual Target
function drawVirtualTarget() {
    const w = vTarget.width;
    const h = vTarget.height;
    const cx = w / 2;
    const cy = h / 2;
    // The max physical radius we draw is the 1-point ring: 23.0mm
    // Scale so 23.0mm fits nicely in the canvas (leaving a 10px margin)
    const scale = (w / 2 - 10) / 23.0;

    vCtx.clearRect(0, 0, w, h);

    // Background (White area)
    vCtx.fillStyle = '#f0f0f5';
    vCtx.fillRect(0, 0, w, h);

    // Draw the black area (4-point ring and inwards)
    // 4-point ring outer radius = 0.5 + (10-4)*2.5 = 15.5mm
    vCtx.beginPath();
    vCtx.arc(cx, cy, 15.5 * scale, 0, 2 * Math.PI);
    vCtx.fillStyle = '#111';
    vCtx.fill();

    // Draw rings 1 to 10
    vCtx.lineWidth = 1;
    for (let i = 1; i <= 10; i++) {
        let r_mm = (i === 10) ? 0.5 : 0.5 + (10 - i) * 2.5;
        vCtx.beginPath();
        vCtx.arc(cx, cy, r_mm * scale, 0, 2 * Math.PI);
        // Rings in the black area (4 and above) use white lines, outer use black lines
        vCtx.strokeStyle = (i >= 4) ? '#fff' : '#000';
        vCtx.stroke();
    }

    // Center dot (just a tiny visual anchor)
    vCtx.beginPath();
    vCtx.arc(cx, cy, 1.5, 0, 2 * Math.PI);
    vCtx.fillStyle = '#fff';
    vCtx.fill();

    // Draw recent shots
    shotHistory.forEach((shot, index) => {
        const isLatest = index === shotHistory.length - 1;
        let r_mm = getRadiusFromScore(shot.score);
        let px = cx + r_mm * scale * Math.cos(shot.angle);
        let py = cy + r_mm * scale * Math.sin(shot.angle);

        vCtx.beginPath();
        // Pellet size representation (visually scaled down a bit to see groupings easily)
        vCtx.arc(px, py, isLatest ? 6 : 4, 0, 2 * Math.PI);
        vCtx.fillStyle = isLatest ? '#ff2a2a' : 'rgba(255, 42, 42, 0.4)';
        vCtx.fill();

        if (isLatest) {
            vCtx.shadowColor = '#ff2a2a';
            vCtx.shadowBlur = 10;
            vCtx.fill();
            vCtx.shadowBlur = 0; // reset
        }
    });
}

clearBtn.addEventListener('click', () => {
    if(confirm('Are you sure you want to clear all history?')) {
        shotHistory = [];
        localStorage.removeItem('beamRifleHistory');
        currentScoreEl.innerText = "0.0";
        lastDetectedScore = null;
        updateUI();
    }
});

// Initial draw
updateUI();
