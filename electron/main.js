import { app, BrowserWindow } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, execSync } from 'node:child_process';
import fs from 'node:fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/** @type {BrowserWindow | null} */
let mainWindow = null;

/** @type {import('child_process').ChildProcess | null} */
let backendProcess = null;

const isDev = process.env.VITE_DEV_SERVER_URL !== undefined;

// Get the backend directory path
function getBackendPath() {
  if (isDev) {
    return path.join(__dirname, '..', 'backend');
  }
  // In production, backend is packaged alongside the app
  return path.join(process.resourcesPath, 'backend');
}

// Check if Python is available
function getPythonCommand() {
  const commands = ['python', 'python3', 'py'];
  
  for (const cmd of commands) {
    try {
      execSync(`${cmd} --version`, { stdio: 'ignore' });
      return cmd;
    } catch {
      continue;
    }
  }
  return null;
}

// Install Python dependencies
function installDependencies(pythonCmd, backendPath) {
  const requirementsPath = path.join(backendPath, 'requirements.txt');
  
  if (!fs.existsSync(requirementsPath)) {
    console.log('No requirements.txt found, skipping dependency install');
    return true;
  }
  
  console.log('Installing Python dependencies...');
  
  try {
    execSync(`${pythonCmd} -m pip install -r "${requirementsPath}" --quiet`, {
      cwd: backendPath,
      stdio: 'inherit',
    });
    console.log('Dependencies installed successfully');
    return true;
  } catch (error) {
    console.error('Failed to install dependencies:', error.message);
    return false;
  }
}

// Start the backend server
function startBackend() {
  const backendPath = getBackendPath();
  const mainPyPath = path.join(backendPath, 'main.py');
  
  if (!fs.existsSync(mainPyPath)) {
    console.error('Backend main.py not found at:', mainPyPath);
    return false;
  }
  
  const pythonCmd = getPythonCommand();
  
  if (!pythonCmd) {
    console.error('Python not found. Please install Python 3.8+ and add it to PATH.');
    return false;
  }
  
  console.log(`Using Python: ${pythonCmd}`);
  console.log(`Backend path: ${backendPath}`);
  
  // Install dependencies first
  installDependencies(pythonCmd, backendPath);
  
  // Start the backend process
  console.log('Starting backend server...');
  
  backendProcess = spawn(pythonCmd, ['main.py'], {
    cwd: backendPath,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: true,
  });
  
  backendProcess.stdout.on('data', (data) => {
    console.log(`[Backend] ${data.toString().trim()}`);
  });
  
  backendProcess.stderr.on('data', (data) => {
    console.error(`[Backend] ${data.toString().trim()}`);
  });
  
  backendProcess.on('error', (error) => {
    console.error('Failed to start backend:', error.message);
  });
  
  backendProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
    backendProcess = null;
  });
  
  return true;
}

// Stop the backend server
function stopBackend() {
  if (backendProcess) {
    console.log('Stopping backend server...');
    
    // On Windows, we need to kill the process tree
    if (process.platform === 'win32') {
      try {
        execSync(`taskkill /pid ${backendProcess.pid} /T /F`, { stdio: 'ignore' });
      } catch {
        // Process might already be dead
      }
    } else {
      backendProcess.kill('SIGTERM');
    }
    
    backendProcess = null;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    title: 'Sentry - AI Detection System',
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    backgroundColor: '#0a0a0a',
    show: false,
  });

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  if (isDev) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  // Start backend first
  startBackend();
  
  // Give backend a moment to start, then create window
  setTimeout(() => {
    createWindow();
  }, 1000);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopBackend();
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackend();
});

app.on('quit', () => {
  stopBackend();
});
