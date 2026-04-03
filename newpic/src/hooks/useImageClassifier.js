import { useRef, useState, useCallback, useEffect } from 'react';
import { ImageClassifier } from '../ml/ImageClassifier';
import { BrowserAdapter } from '../ml/adapters/BrowserAdapter';

/**
 * Custom hook for image classifier operations
 * @returns {Object} { classifierRef, train, predict, exportModel, exportData, loading, progress, message, error }
 */
export const useImageClassifier = () => {
  const classifierRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState(null);
  const [validationMetrics, setValidationMetrics] = useState(null);
  const [trainingSummary, setTrainingSummary] = useState(null);

  // Initialize classifier
  useEffect(() => {
    try {
      classifierRef.current = new ImageClassifier({}, new BrowserAdapter());
      setMessage('Classifier initialized');
      setError(null);
    } catch (err) {
      const errorMessage = `Failed to initialize classifier: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage);
    }

    // Cleanup function
    return () => {
      if (classifierRef.current && classifierRef.current.dispose) {
        classifierRef.current.dispose();
        classifierRef.current = null;
      }
    };
  }, []);

  const train = useCallback(async (imageMap, userConfig = {}) => {
    if (!classifierRef.current) {
      setError('Classifier not initialized');
      return false;
    }

    setLoading(true);
    setError(null);
    setProgress(0);
    setMessage('Training model...');
    setValidationMetrics(null);
    setTrainingSummary(null);

    const startTime = Date.now();
    let finalMetrics = {};
    let splitInfo = null;

    try {
      // Extract training config from userConfig and merge with progressCallback
      const trainingConfig = {
        transferModel: userConfig.transferModel,
        learningRate: userConfig.learningRate,
        optimizer: userConfig.optimizer,
        epochs: userConfig.epochs,
        batchSizeFraction: userConfig.batchSizeFraction,
        useValidation: userConfig.useValidation,
        progressCallback: (event) => {
          setMessage(event.message);
          setProgress(event.progress);

          // Capture split information
          if (event.stage === 'splitting' && event.data?.trainSize) {
            splitInfo = {
              trainSize: event.data.trainSize,
              valSize: event.data.valSize
            };
          }

          // Capture validation metrics from epoch end events
          if (event.data?.type === 'epoch') {
            finalMetrics = {
              loss: event.data.loss,
              acc: event.data.acc,
              val_loss: event.data.val_loss,
              val_acc: event.data.val_acc,
              epoch: event.data.epoch
            };

            if (event.data.val_loss !== undefined) {
              setValidationMetrics({
                val_loss: event.data.val_loss,
                val_acc: event.data.val_acc
              });
            }
          }
        }
      };

      await classifierRef.current.train(imageMap, trainingConfig);

      const endTime = Date.now();
      const duration = ((endTime - startTime) / 1000).toFixed(1);

      // Count total samples and classes
      const classes = Object.keys(imageMap);
      const totalSamples = classes.reduce((sum, label) => sum + imageMap[label].length, 0);

      // Create training summary
      const summary = {
        duration,
        totalSamples,
        numClasses: classes.length,
        trainSize: splitInfo?.trainSize || totalSamples,
        valSize: splitInfo?.valSize || 0,
        useValidation: userConfig.useValidation,
        epochs: userConfig.epochs,
        optimizer: userConfig.optimizer,
        learningRate: userConfig.learningRate,
        batchSizeFraction: userConfig.batchSizeFraction,
        finalMetrics
      };

      console.log('📊 Training Summary:', summary);
      setTrainingSummary(summary);

      setMessage('Training complete!');
      setLoading(false);
      return true;
    } catch (err) {
      const errorMessage = `Training failed: ${err.message}`;
      setError(errorMessage);
      setLoading(false);
      console.error(errorMessage);
      return false;
    }
  }, []);

  const predict = useCallback(async (image) => {
    if (!classifierRef.current) {
      setError('Classifier not initialized or not trained');
      return null;
    }

    try {
      const result = await classifierRef.current.predict(image);
      // Return the predictions array
      return result.predictions;
    } catch (err) {
      const errorMessage = `Prediction failed: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage);
      return null;
    }
  }, []);

  const exportModel = useCallback(async () => {
    if (!classifierRef.current) {
      setError('No model to export');
      return;
    }

    try {
      await classifierRef.current.exportModel();
      setMessage('Model exported successfully');
    } catch (err) {
      const errorMessage = `Export failed: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage);
    }
  }, []);

  const exportData = useCallback(async (imageMap) => {
    if (!classifierRef.current) {
      setError('No data to export');
      return;
    }

    try {
      await classifierRef.current.exportData(imageMap);
      setMessage('Data exported successfully');
    } catch (err) {
      const errorMessage = `Data export failed: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage);
    }
  }, []);

  const loadModel = useCallback(async (modelFiles) => {
    if (!classifierRef.current) {
      setError('Classifier not initialized');
      return false;
    }

    try {
      setLoading(true);
      setMessage('Loading model...');
      await classifierRef.current.load(modelFiles);
      setMessage('Model loaded successfully');
      setLoading(false);
      return true;
    } catch (err) {
      const errorMessage = `Failed to load model: ${err.message}`;
      setError(errorMessage);
      setLoading(false);
      console.error(errorMessage);
      return false;
    }
  }, []);

  const trainOnServer = useCallback(async (imageMap, userConfig, serverUrl) => {
    setLoading(true);
    setError(null);
    setProgress(0);
    setMessage('Preparing images...');
    setValidationMetrics(null);
    setTrainingSummary(null);

    try {
      // Categorize each class: server-session URLs vs browser-local (blob/data) URLs.
      // A class is "server" if its first URL starts with the session image prefix.
      const sessionPrefix = `${serverUrl}/api/images/`;
      let sessionId = null;
      const browserClasses = {}; // classes with blob: or data: URLs

      for (const [label, urls] of Object.entries(imageMap)) {
        const firstUrl = urls[0] || '';
        if (firstUrl.startsWith(sessionPrefix)) {
          if (!sessionId) sessionId = firstUrl.slice(sessionPrefix.length).split('/')[0];
          // Server-session classes are read from disk by the server — no data needed
        } else {
          browserClasses[label] = urls;
        }
      }

      let body;
      // Always include the transfer model config so the server uses the same model as the browser
      const transferModel = userConfig.transferModel || classifierRef.current?.config?.transferModel;
      const serverConfig = { ...userConfig, transferModel };

      if (sessionId && Object.keys(browserClasses).length === 0) {
        // All classes from server session — optimized path, no image data sent
        setMessage('Training from server dataset...');
        body = JSON.stringify({ sessionId, config: serverConfig });
      } else {
        // Resolve blob: URLs to base64 data URLs for the browser-local classes
        const resolvedImageMap = {};
        for (const [label, urls] of Object.entries(browserClasses)) {
          resolvedImageMap[label] = await Promise.all(
            urls.map(async (url) => {
              if (!url.startsWith('blob:')) return url; // already a data URL
              const blob = await fetch(url).then(r => r.blob());
              return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = e => resolve(e.target.result);
                reader.onerror = reject;
                reader.readAsDataURL(blob);
              });
            })
          );
        }

        setMessage('Sending data to server...');
        // Yield one frame so the "Sending..." message renders before
        // JSON.stringify blocks the main thread.
        await new Promise(r => setTimeout(r, 0));
        // Mixed case: session classes (read from disk) + browser classes (base64)
        body = JSON.stringify({ sessionId, imageMap: resolvedImageMap, config: serverConfig });
      }

      const res = await fetch(`${serverUrl}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Server error ${res.status}: ${body}`);
      }

      const { jobId } = await res.json();

      let status = {};
      while (status.status !== 'done' && status.status !== 'error') {
        await new Promise(r => setTimeout(r, 1000));
        const pollRes = await fetch(`${serverUrl}/api/status/${jobId}`);
        status = await pollRes.json();
        setProgress(status.progress || 0);
        setMessage(status.message || 'Training...');
      }

      if (status.status === 'error') throw new Error(status.error);

      setMessage('Loading model from server...');
      if (transferModel && classifierRef.current) {
        classifierRef.current.config.transferModel = transferModel;
      }
      const success = await loadModel(status.modelFiles);
      setMessage('Training complete!');
      setLoading(false);
      return success;
    } catch (err) {
      setError(`Server training failed: ${err.message}`);
      setLoading(false);
      return false;
    }
  }, [loadModel]);

  const loadData = useCallback(async (dataFile) => {
    if (!classifierRef.current) {
      setError('Classifier not initialized');
      return false;
    }

    try {
      setLoading(true);
      setMessage('Loading data...');
      await classifierRef.current.loadData(dataFile);
      setMessage('Data loaded successfully');
      setLoading(false);
      return true;
    } catch (err) {
      const errorMessage = `Failed to load data: ${err.message}`;
      setError(errorMessage);
      setLoading(false);
      console.error(errorMessage);
      return false;
    }
  }, []);

  return {
    classifierRef,
    train,
    trainOnServer,
    predict,
    exportModel,
    exportData,
    loadModel,
    loadData,
    loading,
    progress,
    message,
    error,
    validationMetrics,
    trainingSummary
  };
};
