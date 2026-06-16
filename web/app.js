const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send");
const statusEl = document.querySelector("#status");
const chatHistory = [];
const maxHistoryItems = 6;

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    const mode = data.mode === "prag_lora" ? "PRAG LoRA" : data.mode;
    statusEl.textContent = `${mode} · top_k=${data.top_k}`;
  } catch {
    statusEl.textContent = "Không lấy được trạng thái backend";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  sendButton.disabled = true;
  const pending = addMessage("bot", "Đang suy luận...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: chatHistory.slice(-maxHistoryItems) }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }
    pending.textContent = data.answer;
    chatHistory.push({ role: "user", text: question });
    chatHistory.push({ role: "assistant", text: data.answer });
    if (chatHistory.length > maxHistoryItems) {
      chatHistory.splice(0, chatHistory.length - maxHistoryItems);
    }
  } catch (error) {
    pending.textContent = `Lỗi: ${error.message}`;
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

loadStatus();
