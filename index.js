// index.js — WhatsApp bot with FastAPI classification + safe log storage

import makeWASocket, {
  fetchLatestBaileysVersion,
  useMultiFileAuthState,
  DisconnectReason
} from '@whiskeysockets/baileys';
import qrcode from 'qrcode-terminal';
import fs from 'fs';
import axios from 'axios';
import path from 'path';

// 📁 Safe directories
const AUTH_DIR = './auth_info';
const DATA_DIR = './.data';

if (!fs.existsSync(AUTH_DIR)) fs.mkdirSync(AUTH_DIR, { recursive: true });
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

// 📄 Path to log file
const LOG_FILE = path.join(DATA_DIR, 'message_logs.jsonl');
const logStream = fs.createWriteStream(LOG_FILE, { flags: 'a' });

// 🧠 Function to classify a message using FastAPI
async function classifyAndRespond(sock, messageData) {
  try {
    // Send to FastAPI for classification
    const response = await axios.post('http://localhost:8000/classify', {
      timestamp: messageData.timestamp,
      sender: messageData.sender,
      from_number: messageData.from,
      text: messageData.text,
      summary: messageData.summary
    });

    const classified = response.data;

    // Pretty print to terminal
    console.log(JSON.stringify(classified, null, 2));

    // Append classified data to log file
    logStream.write(JSON.stringify(classified) + '\n');

    // ✅ Send classified result back to YOURSELF (or sender)
    const replyText = `Somethings come up::\n\n📂 : *${classified.summary}*\n*`;

    // 🔁 Send to yourself (or change to messageData.from to reply to sender)
    if (classified.urgency == "High"){
    await sock.sendMessage(`${process.env.PHONENUMBER}@s.whatsapp.net`, {
      text: replyText,

    });}

  } catch (err) {
    console.error('❌ Error classifying or sending message:', err.message);
  }
}

async function start() {
  try {
    const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
    const { version } = await fetchLatestBaileysVersion();

    const sock = makeWASocket({
      version,
      auth: state,
      printQRInTerminal: false,
      browser: ['NightObserver', 'ReplyBot', '1.0']
    });

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('connection.update', (update) => {
      const { connection, lastDisconnect, qr } = update;

      if (qr) {
        console.clear();
        qrcode.generate(qr, { small: true });
        console.log('\n📱 Scan this QR in WhatsApp → Linked devices → Link a device\n');
      }

      if (connection === 'open') {
        console.log('✅ Connected to WhatsApp!');
      } else if (connection === 'close') {
        const reason = lastDisconnect?.error?.output?.statusCode;
        console.error('❌ Connection closed. Reason:', reason);
        if (reason !== DisconnectReason.loggedOut) {
          console.log('Reconnecting in 5s...');
          setTimeout(() => start().catch(console.error), 5000);
        } else {
          console.log('Logged out. Delete ./auth_info to re-link.');
        }
      }
    });

    sock.ev.on('messages.upsert', async (msgUpdate) => {
      try {
        if (!msgUpdate || !msgUpdate.messages) return;

        for (const m of msgUpdate.messages) {
          if (!m.message) continue;
          if (m.key?.fromMe) continue; // Ignore your own messages

          const from = m.key.remoteJid;
          const sender = m.pushName || 'Unknown';
          const timestamp = new Date(Number(m.messageTimestamp) * 1000).toISOString();

          let text = '';
          if (m.message.conversation) text = m.message.conversation;
          else if (m.message?.extendedTextMessage?.text)
            text = m.message.extendedTextMessage.text;
          else if (m.message?.imageMessage?.caption)
            text = m.message.imageMessage.caption;
          else text = '[Non-text message]';

          const messageData = {
            timestamp,
            sender,
            from,
            text
          };

          // 🔁 Send to FastAPI and reply to yourself
          await classifyAndRespond(sock, messageData);
        }
      } catch (err) {
        console.error('Error handling message:', err);
      }
    });

    // Catch all
    process.on('uncaughtException', (err) => {
      console.error('Uncaught exception:', err);
    });

    process.on('unhandledRejection', (reason, p) => {
      console.error('Unhandled rejection at:', p, 'reason:', reason);
    });

    console.log('🤖 Bot started. Waiting for QR scan if required...');
  } catch (err) {
    console.error('Fatal error starting bot:', err);
    setTimeout(start, 5000);
  }
}

start();
