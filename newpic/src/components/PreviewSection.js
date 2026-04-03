import React, { useCallback, useEffect, useRef, useState } from 'react';
import Button from 'react-bootstrap/Button';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { useImageMap } from '../contexts/ImageMapContext';
import { useSharedClassifier } from '../contexts/ImageClassifierContext';
import { useSharedWebcam } from '../contexts/WebcamContext';
import { readFilesAsDataURLs } from '../hooks/useFileReader';
import * as tf from '@tensorflow/tfjs';

const PredictionBars = ({ predictions, compact = false }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: compact ? '6px' : '12px' }}>
    {predictions
      .sort((a, b) => b.confidence - a.confidence)
      .map((pred) => (
        <div key={pred.label}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginBottom: compact ? '2px' : '6px',
            fontSize: compact ? '12px' : '14px',
            fontFamily: 'Poppins-Regular'
          }}>
            <span style={{ fontWeight: pred.rank === 0 ? '600' : '400' }}>
              {pred.label}
            </span>
            <span style={{ color: '#666' }}>
              {Math.round(pred.confidence * 100)}%
            </span>
          </div>
          <ProgressBar
            now={pred.confidence * 100}
            variant={pred.rank === 0 ? 'success' : 'secondary'}
            style={{ height: compact ? '5px' : '8px' }}
          />
        </div>
      ))}
  </div>
);

const ImageCard = ({ item }) => (
  <div style={{
    backgroundColor: '#f8f9fa',
    borderRadius: '10px',
    overflow: 'hidden',
    boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
    display: 'flex',
    flexDirection: 'column'
  }}>
    <div style={{ aspectRatio: '1', overflow: 'hidden', backgroundColor: '#000' }}>
      <img
        src={item.dataUrl}
        alt=""
        style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
      />
    </div>
    <div style={{ padding: '10px' }}>
      {item.loading ? (
        <div style={{ color: '#999', fontSize: '12px', textAlign: 'center', padding: '8px 0' }}>
          Predicting…
        </div>
      ) : !item.predictions ? (
        <div style={{ color: '#dc3545', fontSize: '12px', textAlign: 'center', padding: '8px 0' }}>
          Failed
        </div>
      ) : (
        <PredictionBars predictions={item.predictions} compact />
      )}
    </div>
  </div>
);

const PreviewSection = () => {
  const { isTrained, testImage, setTestImage, predictions, setPredictions, setActiveWebcamClass } = useImageMap();
  const { predict } = useSharedClassifier();
  const { streamRef, working, startCamera } = useSharedWebcam();
  const localVideoRef = useRef(null);
  const fileInputRef = useRef(null);
  const [videoReady, setVideoReady] = useState(false);
  const [uploadedImages, setUploadedImages] = useState([]);

  const isGridMode = uploadedImages.length > 0;

  // Release webcam from training classes and start it for preview
  useEffect(() => {
    if (isTrained && !testImage && !isGridMode) {
      setActiveWebcamClass(null);
      startCamera();
    }
  }, [isTrained, testImage, isGridMode, setActiveWebcamClass, startCamera]);

  // Attach the shared stream to this component's video element
  useEffect(() => {
    if (isTrained && !testImage && !isGridMode && localVideoRef.current && streamRef.current) {
      localVideoRef.current.srcObject = streamRef.current;
      setVideoReady(false);
    }
  }, [isTrained, streamRef, testImage, isGridMode, working]);

  const handleVideoLoaded = useCallback(() => {
    setVideoReady(true);
  }, []);

  // Fallback: Check video ready state periodically
  useEffect(() => {
    if (!isTrained || testImage || isGridMode || !localVideoRef.current) return;

    const checkVideoReady = () => {
      if (localVideoRef.current && localVideoRef.current.readyState >= 2) {
        setVideoReady(true);
      }
    };

    checkVideoReady();
    const timer = setTimeout(checkVideoReady, 1000);
    return () => clearTimeout(timer);
  }, [isTrained, streamRef, testImage, isGridMode]);

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

    // Leave grid mode when capturing from webcam
    setUploadedImages([]);

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
      setTestImage(imageString);
    } catch (error) {
      console.error('Capture failed:', error);
    }
  }, [cropImage, setTestImage, videoReady]);

  const handleUpload = useCallback(async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    e.target.value = '';

    // Leave single/webcam mode when uploading
    if (testImage) setTestImage(null);

    try {
      const dataUrls = await readFilesAsDataURLs(files);

      const initialItems = dataUrls.map((dataUrl, i) => ({
        id: `upload-${Date.now()}-${i}`,
        dataUrl,
        predictions: null,
        loading: true
      }));
      setUploadedImages(initialItems);

      // Run predictions in parallel; update each card as its result arrives
      dataUrls.forEach(async (dataUrl, index) => {
        try {
          const results = await predict(dataUrl);
          setUploadedImages(prev => prev.map((img, i) =>
            i === index ? { ...img, predictions: results, loading: false } : img
          ));
        } catch (err) {
          console.error('Prediction failed for image', index, err);
          setUploadedImages(prev => prev.map((img, i) =>
            i === index ? { ...img, loading: false } : img
          ));
        }
      });
    } catch (err) {
      console.error('Failed to load files:', err);
    }
  }, [predict, testImage, setTestImage]);

  const handleClearGrid = useCallback(() => {
    setUploadedImages([]);
    setActiveWebcamClass(null);
    startCamera();
  }, [setActiveWebcamClass, startCamera]);

  // Predict when a single webcam capture is set
  useEffect(() => {
    if (!testImage || !predict || !isTrained) return;

    const runPrediction = async () => {
      try {
        const results = await predict(testImage);
        if (results) setPredictions(results);
      } catch (error) {
        console.error('Prediction failed:', error);
      }
    };

    runPrediction();
  }, [testImage, predict, isTrained, setPredictions]);

  if (!isTrained) {
    return (
      <section style={{ marginBottom: '60px' }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: '500',
          fontFamily: 'Poppins-Regular',
          marginBottom: '20px',
          color: '#1a1a1d'
        }}>
          3. Preview
        </h2>
        <div style={{
          backgroundColor: 'white',
          borderRadius: '12px',
          padding: '48px 32px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          textAlign: 'center',
          color: '#999'
        }}>
          Train your model first to use preview
        </div>
      </section>
    );
  }

  return (
    <section style={{ marginBottom: '60px' }}>
      <h2 style={{
        fontSize: '20px',
        fontWeight: '500',
        fontFamily: 'Poppins-Regular',
        marginBottom: '20px',
        color: '#1a1a1d'
      }}>
        3. Preview
      </h2>

      {isGridMode ? (
        /* ── Grid mode: multiple uploaded images ── */
        <div style={{
          backgroundColor: 'white',
          borderRadius: '12px',
          padding: '24px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          {/* Toolbar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '20px',
            flexWrap: 'wrap',
            gap: '8px'
          }}>
            <span style={{ fontSize: '14px', color: '#666', fontFamily: 'Poppins-Regular' }}>
              {uploadedImages.length} image{uploadedImages.length !== 1 ? 's' : ''}
            </span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <Button
                variant="outline-primary"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
              >
                Upload More
              </Button>
              <Button
                variant="outline-secondary"
                size="sm"
                onClick={handleClearGrid}
              >
                Clear All
              </Button>
            </div>
          </div>

          {/* Image grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
            gap: '16px'
          }}>
            {uploadedImages.map(item => (
              <ImageCard key={item.id} item={item} />
            ))}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleUpload}
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        /* ── Single / webcam mode ── */
        <div style={{
          backgroundColor: 'white',
          borderRadius: '12px',
          padding: '24px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          display: 'grid',
          gridTemplateColumns: '300px 1fr',
          gap: '32px'
        }}>
          {/* Left: Input */}
          <div>
            <div style={{
              backgroundColor: '#000',
              borderRadius: '8px',
              overflow: 'hidden',
              aspectRatio: '1',
              marginBottom: '12px',
              position: 'relative'
            }}>
              {testImage ? (
                <img
                  src={testImage}
                  alt="Test"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
              ) : (
                <>
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
                  {!working && (
                    <div style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      color: 'white',
                      textAlign: 'center',
                      fontSize: '14px'
                    }}>
                      No webcam
                    </div>
                  )}
                </>
              )}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <Button
                variant="primary"
                onClick={handleCapture}
                disabled={!working || !videoReady}
                style={{ width: '100%' }}
              >
                {!videoReady && working ? 'Loading...' : 'Capture'}
              </Button>
              <Button
                variant="outline-primary"
                onClick={() => fileInputRef.current?.click()}
                style={{ width: '100%' }}
              >
                Upload Images
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleUpload}
                style={{ display: 'none' }}
              />
              {testImage && (
                <Button
                  variant="outline-secondary"
                  onClick={() => setTestImage(null)}
                  style={{ width: '100%', fontSize: '14px' }}
                >
                  Clear
                </Button>
              )}
            </div>
          </div>

          {/* Right: Output */}
          <div>
            <h3 style={{
              fontSize: '16px',
              fontWeight: '500',
              marginBottom: '16px',
              fontFamily: 'Poppins-Regular',
              color: '#1a1a1d'
            }}>
              Output
            </h3>

            {!predictions ? (
              <p style={{ color: '#999', fontSize: '14px' }}>
                Capture or upload an image to see predictions
              </p>
            ) : (
              <PredictionBars predictions={predictions} />
            )}
          </div>
        </div>
      )}
    </section>
  );
};

export default PreviewSection;
