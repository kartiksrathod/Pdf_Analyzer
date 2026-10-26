css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    align-items: flex-start
}
.chat-message.user {
    justify-content: flex-end;
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.chat-sources {
  margin: -0.5rem 0 1.25rem 0;
  padding: 0.6rem 1.75rem;
  background-color: #1b1f2a;
  border-left: 3px solid #7a8798;
  border-radius: 0.5rem;
  color: #cfd6e0;
  font-size: 0.85rem;
}
.chat-sources .sources-title {
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.chat-sources ul {
  margin: 0;
  padding-left: 1.1rem;
}
.chat-sources code {
  color: #ffffff;
  background: transparent;
}
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/6134/6134346.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message" style="text-align:right">{{MSG}}</div>
    <div class="avatar">
        <img src="https://png.pngtree.com/png-vector/20190321/ourmid/pngtree-vector-users-icon-png-image_856952.jpg">
    </div>    
    
</div>
'''

# Source list shown under an answer. Only the static markup here is trusted;
# every dynamic value is escaped with html.escape() before {{ITEMS}} is replaced.
sources_template = '''
<div class="chat-sources">
    <div class="sources-title">Sources</div>
    <ul>{{ITEMS}}</ul>
</div>
'''