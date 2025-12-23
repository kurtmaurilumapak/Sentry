// Note: Preload scripts run in a special context where Electron expects
// CommonJS-style modules, so we must use `require` instead of `import`.
// Using `import` here causes "Cannot use import statement outside a module".

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { contextBridge } = require('electron');

// In this simple prototype we don't need custom Electron APIs yet,
// but we keep the bridge ready for future extensions (e.g. file system, config).
contextBridge.exposeInMainWorld('electronAPI', {
  // placeholder for future APIs
});


