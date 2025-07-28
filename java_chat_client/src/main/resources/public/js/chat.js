
const messagesDiv = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');

function addMessage(text, sender) {
    const div = document.createElement('div');
    div.className = 'msg ' + sender;
    div.innerHTML = `<span class="${sender}">${sender === 'user' ? 'You' : 'Bot'}:</span> ${text}`;
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;
    addMessage(text, 'user');
    userInput.value = '';
    addMessage('...', 'bot');
    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: text })
        });
        const data = await resp.json();
        // Remove the '...' bot message
        messagesDiv.removeChild(messagesDiv.lastChild);
        if (data.response) {
            addMessage(data.response, 'bot');
        } else if (data.error) {
            addMessage('[Error: ' + data.error + ']', 'bot');
        } else {
            addMessage('[Unknown response]', 'bot');
        }
    } catch (err) {
        messagesDiv.removeChild(messagesDiv.lastChild);
        addMessage('[Network error]', 'bot');
    }
});