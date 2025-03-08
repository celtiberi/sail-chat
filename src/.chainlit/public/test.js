console.log('TEST.JS LOADED SUCCESSFULLY!');

// Create a visible notification when the script loads
document.addEventListener('DOMContentLoaded', function() {
    // Create and style the notification
    const notification = document.createElement('div');
    notification.textContent = '🚢 TEST.JS LOADED!';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #ff0000;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        z-index: 1000;
        font-weight: bold;
        font-size: 16px;
    `;

    // Add the notification to the page
    document.body.appendChild(notification);

    // Remove the notification after 5 seconds
    setTimeout(() => notification.remove(), 5000);
}); 