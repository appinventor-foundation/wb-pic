import React, { createContext, useContext } from 'react';
import { useImageClassifier } from '../hooks/useImageClassifier';

const ImageClassifierContext = createContext();

export const useSharedClassifier = () => {
  const context = useContext(ImageClassifierContext);
  if (!context) {
    throw new Error('useSharedClassifier must be used within ImageClassifierProvider');
  }
  return context;
};

export const ImageClassifierProvider = ({ children }) => {
  // Create a single instance of the classifier that's shared across all components
  const classifierState = useImageClassifier();

  return (
    <ImageClassifierContext.Provider value={classifierState}>
      {children}
    </ImageClassifierContext.Provider>
  );
};
