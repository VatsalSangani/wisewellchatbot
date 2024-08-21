document.addEventListener('DOMContentLoaded', function() {
    appendMessage('bot', "Hello! I'm Wise Well, your health assistant. How can I help you today?");
});

function sendMessage() {
    const userInput = document.getElementById('user-input').value.trim();

    if (userInput === "") {
        return; // Don't send empty messages
    }

    appendMessage('user', userInput);

    document.getElementById('user-input').value = '';

    // Show the loading spinner
    const generatingMessage = appendGeneratingMessage();

    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Remove the loading spinner
        removeGeneratingMessage(generatingMessage);

        // Display the bot's response
        appendMessage('bot', data.response);
    })
    .catch(error => {
        removeGeneratingMessage(generatingMessage);
        appendMessage('bot', "Sorry, something went wrong.");
        console.error('Error:', error);
    });
}

function appendMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');

    messageElement.className = `message ${sender}-message`;
    messageElement.innerHTML = `
        <div class="content">
            ${message}
            <div class="timestamp">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    chatBox.appendChild(messageElement);

    // Scroll to the new message smoothly
    messageElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function appendGeneratingMessage() {
    const chatBox = document.getElementById('chat-box');
    const generatingElement = document.createElement('div');

    generatingElement.className = 'generating';
    generatingElement.innerHTML = '<div class="spinner"></div>';

    chatBox.appendChild(generatingElement);

    generatingElement.scrollIntoView({ behavior: 'smooth', block: 'end' });

    return generatingElement;
}

function removeGeneratingMessage(generatingElement) {
    generatingElement.remove();
}

function checkEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}
