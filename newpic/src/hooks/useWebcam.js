import { useRef, useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for webcam operations
 * @returns {Object} { videoRef, working, videoWidth, videoHeight, capture, error }
 */
export const useWebcam = () => {
  const videoRef = useRef(null);
  const [working, setWorking] = useState(false);
  const [videoWidth, setVideoWidth] = useState(0);
  const [videoHeight, setVideoHeight] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 224, height: 224 },
          audio: false
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setWorking(true);
        setError(null);
      } catch (err) {
        const errorMessage = `Failed to access webcam: ${err.message}`;
        setError(errorMessage);
        console.error(errorMessage);
      }
    };

    startCamera();

    // Cleanup function to stop camera when component unmounts
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      setWorking(false);
    };
  }, []);

  const onLoadedData = useCallback(() => {
    if (videoRef.current) {
      setVideoWidth(videoRef.current.videoWidth);
      setVideoHeight(videoRef.current.videoHeight);
    }
  }, []);

  const capture = useCallback(() => {
    if (!videoRef.current) return null;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
  }, []);

  return {
    videoRef,
    working,
    videoWidth,
    videoHeight,
    capture,
    onLoadedData,
    error
  };
};
