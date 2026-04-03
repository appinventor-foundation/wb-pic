import React, { createContext, useContext, useRef, useState, useCallback } from 'react';

const WebcamContext = createContext();

export const useSharedWebcam = () => {
  const context = useContext(WebcamContext);
  if (!context) {
    throw new Error('useSharedWebcam must be used within WebcamProvider');
  }
  return context;
};

export const WebcamProvider = ({ children }) => {
  const streamRef = useRef(null);
  const startingRef = useRef(false); // guard without being in dependency arrays
  const [working, setWorking] = useState(false);
  const [error, setError] = useState(null);

  // Safe to call multiple times — no-ops if already running or in progress.
  // Stable reference (empty deps) so it's safe to use in useEffect dep arrays.
  const startCamera = useCallback(async () => {
    if (streamRef.current) return;
    if (startingRef.current) return;

    startingRef.current = true;
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 224, height: 224 },
        audio: false
      });
      streamRef.current = stream;
      setWorking(true);
    } catch (err) {
      const msg = `Failed to access webcam: ${err.message}`;
      setError(msg);
      console.error(msg);
    } finally {
      startingRef.current = false;
    }
  }, []); // stable — never recreated

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setWorking(false);
  }, []);

  const value = { streamRef, working, error, startCamera, stopCamera };

  return (
    <WebcamContext.Provider value={value}>
      {children}
    </WebcamContext.Provider>
  );
};
