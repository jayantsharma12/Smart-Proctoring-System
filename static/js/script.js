document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const calibrateBtn = document.getElementById('calibrate-btn');
    const videoFeed = document.getElementById('video-feed');
    const statusIndicator = document.getElementById('indicator-light');
    const statusText = document.getElementById('status-text');
    const notificationArea = document.getElementById('notification-area');
    const calibrationModal = document.getElementById('calibration-modal');
    const cancelCalibrationBtn = document.getElementById('cancel-calibration');
    const calibrationProgressBar = document.getElementById('calibration-progress-bar');
    const seriousViolationsCount = document.querySelector('#serious-violations .count');
    
    // Violation elements
    const violationElements = {
        'face_missing': document.getElementById('face-missing'),
        'multiple_faces': document.getElementById('multiple-faces'),
        'looking_away': document.getElementById('looking-away'),
        'speaking': document.getElementById('speaking'),
        'mouth_open': document.getElementById('mouth-open')
    };
    
    // State variables
    let proctorRunning = false;
    let violationCheckInterval = null;
    let calibrationInterval = null;
    
    // Start proctoring
    startBtn.addEventListener('click', function() {
        fetch('/start', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                proctorRunning = true;
                showNotification('success', 'Proctoring system started successfully');
                updateUIState();
                
                // Start the video feed
                videoFeed.src = '/video_feed?' + new Date().getTime();
                
                // Start checking violations
                startViolationCheck();
            } else {
                showNotification('error', data.message || 'Failed to start proctoring');
            }
        })
        .catch(error => {
            showNotification('error', 'Network error: ' + error.message);
        });
    });
    
    // Stop proctoring
    stopBtn.addEventListener('click', function() {
        fetch('/stop', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                proctorRunning = false;
                showNotification('success', 'Proctoring system stopped');
                updateUIState();
                
                // Stop checking violations
                stopViolationCheck();
                
                // Clear the video feed
                videoFeed.src = '';
            } else {
                showNotification('error', data.message || 'Failed to stop proctoring');
            }
        })
        .catch(error => {
            showNotification('error', 'Network error: ' + error.message);
        });
    });
    
    // Calibrate mouth
    calibrateBtn.addEventListener('click', function() {
        if (!proctorRunning) {
            showNotification('error', 'Please start the proctoring system first');
            return;
        }
        
        // Show calibration modal
        calibrationModal.classList.add('active');
        
        // Fake progress animation for user feedback
        let progress = 0;
        calibrationInterval = setInterval(() => {
            progress += 3.33; // 30 frames at 100ms = 3 seconds
            calibrationProgressBar.style.width = progress + '%';
            
            if (progress >= 100) {
                clearInterval(calibrationInterval);
                
                // Submit calibration request
                fetch('/calibrate_mouth', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showNotification('success', 'Mouth calibration completed successfully');
                    } else {
                        showNotification('error', data.message || 'Failed to calibrate mouth');
                    }
                    calibrationModal.classList.remove('active');
                })
                .catch(error => {
                    showNotification('error', 'Network error: ' + error.message);
                    calibrationModal.classList.remove('active');
                });
            }
        }, 100);
    });
    
    // Cancel calibration
    cancelCalibrationBtn.addEventListener('click', function() {
        clearInterval(calibrationInterval);
        calibrationModal.classList.remove('active');
        calibrationProgressBar.style.width = '0%';
    });
    
    // Update UI based on current state
    function updateUIState() {
        if (proctorRunning) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusIndicator.classList.add('active');
            statusText.textContent = 'Running';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusIndicator.classList.remove('active');
            statusText.textContent = 'Not Running';
            
            // Reset violation displays
            resetViolationDisplay();
        }
    }
    
    // Start checking violations periodically
    function startViolationCheck() {
        violationCheckInterval = setInterval(checkViolations, 1000);
    }
    
    // Stop checking violations
    function stopViolationCheck() {
        if (violationCheckInterval) {
            clearInterval(violationCheckInterval);
            violationCheckInterval = null;
        }
    }
    
    // Check for violations
    function checkViolations() {
        fetch('/get_violations')
            .then(response => response.json())
            .then(data => {
                updateViolationDisplay(data.violations, data.thresholds, data.serious_count);
            })
            .catch(error => {
                console.error('Error fetching violations:', error);
            });
    }
    
    // Update violation display
    function updateViolationDisplay(violations, thresholds, seriousCount) {
        // Update serious violations count
        seriousViolationsCount.textContent = seriousCount;
        
        // Update each violation item
        Object.keys(violations).forEach(key => {
            const element = violationElements[key];
            if (!element) return;
            
            const count = violations[key];
            const threshold = thresholds[key];
            const percentage = Math.min((count / threshold) * 100, 100);
            
            // Update progress bar
            const progressBar = element.querySelector('.progress-bar');
            progressBar.style.width = percentage + '%';
            
            // Update count text
            const countText = element.querySelector('.violation-count');
            countText.textContent = count + '/' + threshold;
            
            // Highlight if threshold met
            if (count >= threshold) {
                element.classList.add('active');
            } else {
                element.classList.remove('active');
            }
        });
    }
    
    // Reset violation display
    function resetViolationDisplay() {
        Object.values(violationElements).forEach(element => {
            const progressBar = element.querySelector('.progress-bar');
            progressBar.style.width = '0%';
            
            const countText = element.querySelector('.violation-count');
            countText.textContent = '0/' + countText.textContent.split('/')[1];
            
            element.classList.remove('active');
        });
        
        seriousViolationsCount.textContent = '0';
    }
    
    // Show notification
    function showNotification(type, message) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div>${message}</div>
            <button class="notification-close">&times;</button>
        `;
        
        notification.querySelector('.notification-close').addEventListener('click', function() {
            notification.remove();
        });
        
        notificationArea.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
    
    // Initial UI state
    updateUIState();
});