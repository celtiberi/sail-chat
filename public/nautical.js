// Nautical Theme JavaScript
console.log('nautical.js loading...');

// Initialize Google Analytics
if (!window.gtag) {  // Only initialize if not already present
    console.log('Initializing Google Analytics...');

    // Create and inject the gtag.js script
    const gtagScript = document.createElement('script');
    gtagScript.async = true;
    gtagScript.src = 'https://www.googletagmanager.com/gtag/js?id=G-WQQYFQL32V';
    document.head.appendChild(gtagScript);

    // Initialize gtag
    window.dataLayer = window.dataLayer || [];
    window.gtag = function () { dataLayer.push(arguments); }
    gtag('js', new Date());
    gtag('config', 'G-WQQYFQL32V');

    console.log('Google Analytics initialized');
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded fired in nautical.js');
    
    
    // Add animations to the document
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
        }
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Add the notification to the page
    document.body.appendChild(notification);

    // Remove the notification after animation
    setTimeout(() => notification.remove(), 3500);

    // Initialize Google Analytics
    if (!window.gtag) {  // Only initialize if not already present
        console.log('Initializing Google Analytics...');
        
        // Create and inject the gtag.js script
        const gtagScript = document.createElement('script');
        gtagScript.async = true;
        gtagScript.src = 'https://www.googletagmanager.com/gtag/js?id=G-WQQYFQL32V';
        document.head.appendChild(gtagScript);

        // Initialize gtag
        window.dataLayer = window.dataLayer || [];
        window.gtag = function() { dataLayer.push(arguments); }
        gtag('js', new Date());
        gtag('config', 'G-WQQYFQL32V');
        
        console.log('Google Analytics initialized');
    }

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