import { ipcMain } from 'electron'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

let worker: ChildProcessWithoutNullStreams | null = null
let isTranscribing = false
let workerReady = false
let purpose: TranscriptionPurpose = 'screenshot'
let accumulatedText = ''
let currentPartial = ''
let inputBuffer = ''

export type TranscriptionPurpose = 'screenshot' | 'conversation'
export interface TranscriptionOptions { purpose?: TranscriptionPurpose; maxSentenceSilence?: number }
type SentenceListener = (text: string, final: boolean) => void
const sentenceListeners = new Set<SentenceListener>()
const endListeners = new Set<() => void>()

export function onConversationSentence(listener: SentenceListener): void { sentenceListeners.add(listener) }
export function onTranscriptionEnd(listener: () => void): void { endListeners.add(listener) }
export function isTranscriptionRunning(wanted: TranscriptionPurpose): boolean { return isTranscribing && purpose === wanted }
function sendToRenderer(channel: string, ...args: unknown[]) {
  const mainWindow = global.mainWindow
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(channel, ...args)
}
function reportStopped() { sendToRenderer('transcription-stopped'); endListeners.forEach((listener) => listener()) }
function resolvePython(): string {
  const candidates = [
    process.env.INTERVIEW_CODER_PYTHON,
    join(homedir(), 'Documents/Codex/2026-10-05/ni/work/jarvis-venv/bin/python'),
    join(homedir(), 'fujian-social-insurance-assistant/.venv-voice/bin/python'),
    'python3'
  ].filter(Boolean) as string[]
  return candidates.find((p) => p === 'python3' || existsSync(p)) || 'python3'
}
function resolveModel(): string | null {
  const candidates = [
    process.env.INTERVIEW_CODER_WHISPER_MODEL,
    join(homedir(), 'jarvis-codex/models/whisper'),
    join(homedir(), 'fujian-social-insurance-assistant/data/voice-models/whisper-small')
  ].filter(Boolean) as string[]
  return candidates.find((p) => existsSync(p)) || null
}
function getWorkerScript(): string {
  const bundled = join(process.resourcesPath, 'scripts', 'local_asr_worker.py')
  if (existsSync(bundled)) return bundled
  return join(fileURLToPath(new URL('.', import.meta.url)), '../../scripts/local_asr_worker.py')
}
function cleanup() {
  if (worker && !worker.killed) worker.kill('SIGTERM')
  worker = null; workerReady = false; isTranscribing = false; inputBuffer = ''
}
function handleWorkerLine(line: string) {
  try {
    const msg = JSON.parse(line) as { type: string; text?: string; final?: boolean; message?: string }
    if (msg.type === 'ready') { workerReady = true; return }
    if (msg.type === 'error') { sendToRenderer('transcription-error', msg.message || '本地语音识别失败'); return }
    if (msg.type !== 'result' || !msg.text) return
    const text = msg.text
    if (purpose === 'conversation') { sentenceListeners.forEach((listener) => listener(text, msg.final !== false)); return }
    if (msg.final !== false) { accumulatedText += text; currentPartial = '' } else currentPartial = text
    sendToRenderer('transcription-text', { text: getTranscriptionText(), isPartial: msg.final === false })
  } catch (err) { console.error('Failed to parse local ASR result:', err) }
}
function startTranscription(options: TranscriptionOptions = {}) {
  if (isTranscribing) return
  const model = resolveModel()
  if (!model) { sendToRenderer('transcription-error', '找不到本地 Whisper 模型。请设置 INTERVIEW_CODER_WHISPER_MODEL。'); return }
  purpose = options.purpose ?? 'screenshot'; accumulatedText = ''; currentPartial = ''
  worker = spawn(resolvePython(), [getWorkerScript(), '--model', model], { stdio: ['pipe', 'pipe', 'pipe'] })
  isTranscribing = true
  worker.stdout.setEncoding('utf8')
  worker.stdout.on('data', (data: string) => { inputBuffer += data; const lines = inputBuffer.split('\n'); inputBuffer = lines.pop() || ''; lines.filter(Boolean).forEach(handleWorkerLine) })
  worker.stderr.on('data', (data) => console.error('[local-asr]', data.toString()))
  worker.on('error', (err) => { sendToRenderer('transcription-error', `本地 Whisper 启动失败：${err.message}`); cleanup(); reportStopped() })
  worker.on('exit', () => { if (isTranscribing) { cleanup(); reportStopped() } })
}
function stopTranscription() { if (!isTranscribing) return; if (worker?.stdin.writable) worker.stdin.write(JSON.stringify({ type: 'stop' }) + '\n'); cleanup(); reportStopped() }
function handleAudioChunk(chunk: ArrayBuffer) { if (!workerReady || !worker?.stdin.writable) return; worker.stdin.write(JSON.stringify({ type: 'audio', pcm16: Buffer.from(chunk).toString('base64') }) + '\n') }
export function getTranscriptionText(): string { return accumulatedText + currentPartial }
export function clearTranscriptionText() { accumulatedText = ''; currentPartial = '' }
ipcMain.handle('start-transcription', (_event, options?: TranscriptionOptions) => startTranscription(options))
ipcMain.handle('stop-transcription', () => stopTranscription())
ipcMain.on('transcription-audio-chunk', (_event, chunk: ArrayBuffer) => handleAudioChunk(chunk))
ipcMain.handle('get-transcription-text', () => getTranscriptionText())
ipcMain.handle('clear-transcription-text', () => clearTranscriptionText())
