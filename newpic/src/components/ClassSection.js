import React, { useCallback, useRef, useEffect, useState } from 'react';
import Button from 'react-bootstrap/Button';
import { useImageMap } from '../contexts/ImageMapContext';
import { useSharedWebcam } from '../contexts/WebcamContext';
import { readFilesAsDataURLs } from '../hooks/useFileReader';
import * as tf from '@tensorflow/tfjs';

const ClassSection = ({ className }) => {
  const { imageMap, addImage, removeImage, removeLabel, activeWebcamClass, setActiveWebcamClass } = useImageMap();
  const images = imageMap[className] || [];
  const localVideoRef = useRef(null);
  const { streamRef, working, error: webcamError, startCamera } = useSharedWebcam();
  const [videoReady, setVideoReady] = useState(false);
  const recordingIntervalRef = useRef(null);

  const isWebcamActive = activeWebcamClass === className;

  // Start the camera whenever this class's webcam section becomes visible.
  // This covers both toggleWebcam() and addLabel() auto-activate paths.
  useEffect(() => {
    if (isWebcamActive) {
      startCamera();
    }
  }, [isWebcamActive, startCamera]);

  // Attach the stream once the camera is ready AND this class is active.
  useEffect(() => {
    if (isWebcamActive && working && localVideoRef.current && streamRef.current) {
      localVideoRef.current.srcObject = streamRef.current;
      setVideoReady(false);
    } else if (!isWebcamActive && localVideoRef.current) {
      localVideoRef.current.srcObject = null;
      setVideoReady(false);
    }
  }, [isWebcamActive, working, streamRef]);

  const handleVideoLoaded = useCallback(() => {
    setVideoReady(true);
  }, []);

  const toggleWebcam = useCallback(() => {
    setActiveWebcamClass(isWebcamActive ? null : className);
    // startCamera() is called by the isWebcamActive effect above
  }, [isWebcamActive, className, setActiveWebcamClass]);

  // Fallback: Check video ready state periodically when webcam is active
  useEffect(() => {
    if (!isWebcamActive || !localVideoRef.current) return;

    const checkVideoReady = () => {
      if (localVideoRef.current && localVideoRef.current.readyState >= 2) {
        setVideoReady(true);
      }
    };

    // Check immediately
    checkVideoReady();

    // Check again after a delay in case loadeddata event didn't fire
    const timer = setTimeout(checkVideoReady, 1000);

    return () => clearTimeout(timer);
  }, [isWebcamActive, streamRef]);

  const cropImage = useCallback((img) => {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }, []);

  const handleCapture = useCallback(async () => {
    if (!localVideoRef.current || !videoReady) {
      console.warn('Video not ready for capture');
      return;
    }

    try {
      const result = tf.tidy(() => {
        const webcamImage = tf.browser.fromPixels(localVideoRef.current);
        const croppedImage = tf.keep(cropImage(webcamImage));
        return croppedImage;
      });

      let newImage = await tf.browser.toPixels(result);
      let imageData = new ImageData(newImage, result.shape[0]);
      let canvas = document.createElement('canvas');
      let ctx = canvas.getContext('2d');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      ctx.putImageData(imageData, 0, 0);
      let imageString = canvas.toDataURL();

      result.dispose();
      addImage(className, imageString);
    } catch (error) {
      console.error('Capture failed:', error);
    }
  }, [cropImage, className, addImage, videoReady]);

  const handleUpload = useCallback(async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      try {
        const dataUrls = await readFilesAsDataURLs(files);
        dataUrls.forEach(url => addImage(className, url));
      } catch (err) {
        console.error('Failed to load files:', err);
      }
    }
    e.target.value = '';
  }, [className, addImage]);

  const handleRemoveImage = useCallback((image) => {
    removeImage(className, image);
  }, [className, removeImage]);

  const startRecording = useCallback(() => {
    if (!videoReady || !working) return;

    // Capture immediately
    handleCapture();

    // Then capture every 200ms while holding
    recordingIntervalRef.current = setInterval(() => {
      handleCapture();
    }, 200);
  }, [handleCapture, videoReady, working]);

  const stopRecording = useCallback(() => {
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
  }, []);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    };
  }, []);

  return (
    <div style={{
      backgroundColor: 'white',
      borderRadius: '12px',
      padding: '24px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      display: 'grid',
      gridTemplateColumns: '250px 1fr',
      gap: '24px',
      alignItems: 'start',
      position: 'relative'
    }}>
      {/* Delete Class Button - Top Right of Card */}
      <button
        onClick={() => removeLabel(className)}
        style={{
          position: 'absolute',
          top: '12px',
          right: '12px',
          background: 'transparent',
          border: '1px solid transparent',
          borderRadius: '6px',
          cursor: 'pointer',
          fontSize: '14px',
          color: '#999',
          padding: '4px 6px',
          lineHeight: '1',
          zIndex: 10,
          transition: 'all 0.2s'
        }}
        onMouseEnter={(e) => {
          e.target.style.background = '#fee';
          e.target.style.borderColor = '#fcc';
          e.target.style.color = '#c33';
        }}
        onMouseLeave={(e) => {
          e.target.style.background = 'transparent';
          e.target.style.borderColor = 'transparent';
          e.target.style.color = '#999';
        }}
        title="Delete this class"
        aria-label={`Delete ${className} class`}
      >
        🗑️
      </button>

      {/* Left: Class name and webcam */}
      <div>
        <h3 style={{
          fontSize: '18px',
          fontWeight: '500',
          margin: 0,
          marginBottom: '12px',
          fontFamily: 'Poppins-Regular',
          color: '#1a1a1d',
          lineHeight: '1.5',
          minHeight: '27px'
        }}>
          {className}
        </h3>

        {/* Webcam - only show if active */}
        {isWebcamActive ? (
          <>
            <div style={{
              backgroundColor: '#000',
              borderRadius: '8px',
              overflow: 'hidden',
              position: 'relative',
              aspectRatio: '1',
              marginBottom: '12px',
              border: '2px solid #000'
            }}>
              <video
                ref={localVideoRef}
                autoPlay
                playsInline
                muted
                onLoadedData={handleVideoLoaded}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: working ? 'block' : 'none'
                }}
              />
              {!working && webcamError && (
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  color: '#f88',
                  textAlign: 'center',
                  fontSize: '13px',
                  padding: '20px'
                }}>
                  No webcam found
                </div>
              )}
            </div>

            {/* Capture & Upload Buttons */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <Button
                variant="primary"
                onPointerDown={startRecording}
                onPointerUp={stopRecording}
                onPointerLeave={stopRecording}
                disabled={!working || !videoReady}
                style={{ width: '100%' }}
              >
                {!videoReady && working ? 'Loading...' : 'Hold to Record'}
              </Button>
              {/* label directly activates the input — no programmatic .click() */}
              <label style={{
                display: 'block',
                width: '100%',
                padding: '6px 12px',
                fontSize: '14px',
                fontFamily: 'Poppins-Regular',
                textAlign: 'center',
                color: '#0d6efd',
                border: '1px solid #0d6efd',
                borderRadius: '4px',
                cursor: 'pointer',
                userSelect: 'none',
                boxSizing: 'border-box'
              }}>
                Upload Images
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleUpload}
                  style={{ display: 'none' }}
                />
              </label>
              <Button
                variant="outline-secondary"
                onClick={toggleWebcam}
                style={{ width: '100%', fontSize: '14px' }}
              >
                Hide Webcam
              </Button>
            </div>
          </>
        ) : (
          <Button
            variant="outline-primary"
            onClick={toggleWebcam}
            style={{ width: '100%', marginBottom: '12px' }}
          >
            📷 Show Webcam
          </Button>
        )}
      </div>

      {/* Right: Sample images */}
      <div>
        <div style={{
          fontSize: '14px',
          color: '#666',
          marginBottom: '12px',
          fontFamily: 'Poppins-Regular',
          lineHeight: '1.5',
          minHeight: '27px',
          display: 'flex',
          alignItems: 'center'
        }}>
          {images.length} {images.length === 1 ? 'sample' : 'samples'}
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
          gap: '12px',
          maxHeight: '250px',
          overflowY: 'auto',
          padding: '8px',
          border: '2px solid #ddd',
          borderRadius: '8px',
          position: 'relative',
          boxShadow: 'inset 0 -20px 20px -20px rgba(0,0,0,0.3)',
          background: 'linear-gradient(white 20%, rgba(255,255,255,0.8) 50%, rgba(255,255,255,0)) center top / 100% 60px no-repeat, linear-gradient(rgba(255,255,255,0), rgba(255,255,255,0.8) 50%, white 80%) center bottom / 100% 60px no-repeat, radial-gradient(farthest-side at 50% 0, rgba(0,0,0,0.3), transparent) center top / 100% 15px no-repeat, radial-gradient(farthest-side at 50% 100%, rgba(0,0,0,0.3), transparent) center bottom / 100% 15px no-repeat',
          backgroundAttachment: 'local, local, scroll, scroll'
        }}>
          {images.length === 0 ? (
            <div style={{
              gridColumn: '1 / -1',
              textAlign: 'center',
              padding: '40px 20px',
              color: '#999',
              fontSize: '14px',
              border: '2px dashed #ddd',
              borderRadius: '8px'
            }}>
              Add samples using webcam or upload
            </div>
          ) : (
            <>
              {images.slice(0, 30).map((img, idx) => (
                <div
                  key={idx}
                  style={{
                    position: 'relative',
                    aspectRatio: '1',
                    borderRadius: '6px',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    border: '1px solid #eee'
                  }}
                  onMouseEnter={(e) => {
                    const btn = e.currentTarget.querySelector('.delete-btn');
                    if (btn) btn.style.opacity = '1';
                  }}
                  onMouseLeave={(e) => {
                    const btn = e.currentTarget.querySelector('.delete-btn');
                    if (btn) btn.style.opacity = '0';
                  }}
                >
                  <img
                    src={img}
                    alt={`Sample ${idx + 1}`}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover'
                    }}
                  />
                  <button
                    className="delete-btn"
                    onClick={() => handleRemoveImage(img)}
                    style={{
                      position: 'absolute',
                      top: '4px',
                      right: '4px',
                      backgroundColor: 'rgba(0,0,0,0.7)',
                      color: 'white',
                      border: 'none',
                      borderRadius: '50%',
                      width: '24px',
                      height: '24px',
                      cursor: 'pointer',
                      fontSize: '16px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: '0',
                      transition: 'opacity 0.2s'
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}
              {images.length > 30 && (
                <div style={{
                  aspectRatio: '1',
                  borderRadius: '6px',
                  border: '1px solid #eee',
                  backgroundColor: '#f8f9fa',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '13px',
                  color: '#666',
                  fontFamily: 'Poppins-Regular',
                  textAlign: 'center',
                  padding: '4px'
                }}>
                  +{images.length - 30} more
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ClassSection;
