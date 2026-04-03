import React, { createContext, useContext, useState, useCallback } from 'react';

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
  const [serverUrl, setServerUrl] = useState('http://localhost:5000');

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
    setServerUrl
  };

  return (
    <ImageMapContext.Provider value={value}>
      {children}
    </ImageMapContext.Provider>
  );
};
