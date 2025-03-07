// Nautical Theme JavaScript

// Google Analytics
(function () {
    // Add Google Analytics script
    const gaScript = document.createElement('script');
    gaScript.async = true;
    gaScript.src = 'https://www.googletagmanager.com/gtag/js?id=G-WQQYFQL32V';
    document.head.appendChild(gaScript);

    // Initialize Google Analytics
    window.dataLayer = window.dataLayer || [];
    function gtag() { dataLayer.push(arguments); }
    gtag('js', new Date());
    gtag('config', 'G-WQQYFQL32V');
})();

// Add wave animation to the background
document.addEventListener('DOMContentLoaded', function() {
    // Create wave animation container
    const waveContainer = document.createElement('div');
    waveContainer.className = 'wave-container';
    document.body.appendChild(waveContainer);

    // Add wave elements
    for (let i = 0; i < 3; i++) {
        const wave = document.createElement('div');
        wave.className = `wave wave-${i + 1}`;
        waveContainer.appendChild(wave);
    }

    // Add compass animation to loading indicator
    const loadingIndicator = document.querySelector('.loading');
    if (loadingIndicator) {
        loadingIndicator.classList.add('compass');
    }

    // Add nautical-themed tooltips
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'nautical-tooltip';
            tooltip.textContent = this.getAttribute('data-tooltip');
            this.appendChild(tooltip);
        });

        element.addEventListener('mouseleave', function() {
            const tooltip = this.querySelector('.nautical-tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });

    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Add nautical-themed message animations
function addMessageAnimation(message) {
    message.classList.add('message-animation');
    setTimeout(() => {
        message.classList.remove('message-animation');
    }, 1000);
}

// Add nautical-themed file upload handling
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const fileType = file.type;
        if (fileType.startsWith('image/')) {
            // Add image preview with nautical frame
            const preview = document.createElement('div');
            preview.className = 'nautical-image-preview';
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            preview.appendChild(img);
            document.querySelector('.file-upload').appendChild(preview);
        }
    }
}

// Add nautical-themed error handling
function showError(message) {
    const errorContainer = document.createElement('div');
    errorContainer.className = 'nautical-error';
    errorContainer.textContent = message;
    document.body.appendChild(errorContainer);
    
    setTimeout(() => {
        errorContainer.remove();
    }, 5000);
}

// Add nautical-themed success messages
function showSuccess(message) {
    const successContainer = document.createElement('div');
    successContainer.className = 'nautical-success';
    successContainer.textContent = message;
    document.body.appendChild(successContainer);
    
    setTimeout(() => {
        successContainer.remove();
    }, 3000);
} 