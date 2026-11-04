import React, { createContext, useContext, useState, useCallback } from 'react';
import { SERVER_URL } from '../config/runtime';

const ImageMapContext = createContext();

export const useImageMap = () => {
  const context = useContext(ImageMapContext);
  if (!context) {
    throw new Error('useImageMap must be used within ImageMapProvider');
  }
  return context;
};

export const ImageMapProvider = ({ children }) => {
  const [imageMap, setImageMap] = useState({});
  const [loadedModel, setLoadedModel] = useState(undefined);
  const [isTrained, setIsTrained] = useState(false);
  const [testImage, setTestImage] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [activeWebcamClass, setActiveWebcamClass] = useState(null);
  const [serverUrl, setServerUrl] = useState(SERVER_URL);
  // Inference history, grouped by training run: [{ id, label, trainedAt, entries: [...] }]
  const [history, setHistory] = useState([]);

  const startRun = useCallback((label) => {
    setHistory(prev => {
      const number = (prev.length ? prev[prev.length - 1].number : 0) + 1;
      return [...prev, { id: `run-${Date.now()}`, number, label, trainedAt: Date.now(), entries: [] }];
    });
  }, []);

  const addHistoryEntry = useCallback((entry) => {
    setHistory(prev => {
      // Inference without a training run in this session (e.g. a loaded model)
      const runs = prev.length
        ? prev
        : [{ id: `run-${Date.now()}`, number: 1, label: 'Loaded model', trainedAt: Date.now(), entries: [] }];
      const current = runs[runs.length - 1];
      return [...runs.slice(0, -1), { ...current, entries: [...current.entries, entry] }];
    });
  }, []);

  // Drops past runs and entries but keeps the current run so new inferences stay attributed to it
  const clearHistory = useCallback(() => {
    setHistory(prev => (prev.length ? [{ ...prev[prev.length - 1], entries: [] }] : []));
  }, []);

  const addLabel = useCallback((labelName) => {
    setImageMap(prev => ({
      ...prev,
      [labelName]: []
    }));
    // Automatically move webcam to newly created class
    setActiveWebcamClass(labelName);
  }, []);

  const removeLabel = useCallback((labelName) => {
    setImageMap(prev => {
      const newMap = { ...prev };
      delete newMap[labelName];
      return newMap;
    });
  }, []);

  const addImage = useCallback((labelName, image) => {
    setImageMap(prev => ({
      ...prev,
      [labelName]: [...(prev[labelName] || []), image]
    }));
  }, []);

  const removeImage = useCallback((labelName, imageToRemove) => {
    setImageMap(prev => {
      const images = [...prev[labelName]];
      const index = images.indexOf(imageToRemove);
      if (index !== -1) {
        images.splice(index, 1);
      }
      return {
        ...prev,
        [labelName]: images
      };
    });
  }, []);

  const updateImageMap = useCallback((newImageMap) => {
    setImageMap(newImageMap);
  }, []);

  const value = {
    imageMap,
    loadedModel,
    setLoadedModel,
    isTrained,
    setIsTrained,
    testImage,
    setTestImage,
    predictions,
    setPredictions,
    activeWebcamClass,
    setActiveWebcamClass,
    addLabel,
    removeLabel,
    addImage,
    removeImage,
    updateImageMap,
    serverUrl,
    setServerUrl,
    history,
    startRun,
    addHistoryEntry,
    clearHistory
  };

  return (
    <ImageMapContext.Provider value={value}>
      {children}
    </ImageMapContext.Provider>
  );
};
