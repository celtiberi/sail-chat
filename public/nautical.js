// Nautical Theme JavaScript
console.log('nautical.js loading... v1.1.0'); // Added version number for tracking

// Cache-busting mechanism
(function() {
    // Check if this is a new version that needs cache busting
    const currentVersion = '1.1.1';
    const storedVersion = localStorage.getItem('nauticalVersion');
    
    if (storedVersion !== currentVersion) {
        console.log('New version detected, clearing cache data...');
        // Clear any cached data
        localStorage.removeItem('nauticalGuideShown');
        // Store the new version
        localStorage.setItem('nauticalVersion', currentVersion);
        
        // Force reload if this isn't the initial page load
        if (storedVersion) {
            console.log('Reloading page to apply updates...');
            // Add a small delay to ensure logging completes
            setTimeout(() => {
                window.location.reload(true);
            }, 100);
        }
    }
})();

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

// Add user guide when the DOM is fully loaded
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
        .nautical-user-guide {
            animation: slideIn 0.5s ease-out;
        }
    `;
    document.head.appendChild(style);
    
    // Create user guide banner
    const userGuide = document.createElement('div');
    userGuide.className = 'nautical-user-guide';
    userGuide.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(to right, #1a3a5f, #2c5f8e);
        color: white;
        padding: 15px 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        font-size: 14px;
        line-height: 1.5;
        border-left: 5px solid #4a90e2;
        max-width: 400px;
        overflow: hidden;
        z-index: 1000;
    `;
    
    // Add content to the user guide
    userGuide.innerHTML = `
        <h3 style="margin-top: 0; color: #4a90e2; font-size: 18px;">
            <i class="fa fa-compass" style="margin-right: 8px;"></i>Welcome to the Sailor's Parrot
        </h3>
        <p>About this chatbot:</p>
        <ul style="padding-left: 20px; margin-bottom: 10px;">
            <li>This is not a quick chatbot. It uses many sources and AI calls to answer your question. Give it time to respond.</li>
            <li>It's a work in progress and will improve over time.</li>
        </ul>
        <p>Here's how to get the most out of your sailing companion:</p>
        <ul style="padding-left: 20px; margin-bottom: 10px;">
            <li>Ask specific questions about sailing techniques, navigation, or boat and engine maintenance</li>            
        </ul>
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <span>Example: "What safety equipment should I have for coastal sailing in the Pacific Northwest?"</span>
            <span>Example: "What is the best route from Isla Mujeres to Key West?"</span>
            <span>Example: "Water is not flowing through my engine very well.  What are the possible causes?"</span>
            
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #4a90e2;">
                <h4 style="margin-top: 0; color: #4a90e2; font-size: 16px;">🆕 Recent Updates:</h4>
                <ul style="padding-left: 20px; margin-bottom: 10px;">
                    <li><strong>DeepSeek Integration:</strong> We've integrated with DeepSeek for enhanced reasoning responses, providing more detailed and accurate nautical information.</li>
                    <li>Improved response times for complex navigation questions</li>
                    <li>Enhanced technical knowledge about boat maintenance and repairs</li>
                    <li>Automatic updates - no need to clear your cache!</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 5px;">
                <button id="close-guide" style="background: #4a90e2; border: none; color: white; padding: 5px 10px; border-radius: 4px; cursor: pointer;">
                    Got it!
                </button>
            </div>
        </div>
    `;
    
    // Always show the guide (removed localStorage check)
    
    // Add the guide to the page
    document.body.appendChild(userGuide);
    console.log('User guide added to page');
    
    // Add event listener to close button
    document.getElementById('close-guide').addEventListener('click', function() {
        console.log('Close guide button clicked');
        userGuide.style.animation = 'fadeOut 0.5s forwards';
        setTimeout(() => {
            userGuide.remove();
        }, 500);
        
        // Remove localStorage setting so it shows again next time
        localStorage.removeItem('nauticalGuideShown');
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