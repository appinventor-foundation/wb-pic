import { useState, useCallback } from 'react';
import { readFilesAsDataURLs } from './useFileReader';

/**
 * Custom hook for handling file drag and drop operations
 * @param {string} baseClassName - Base CSS class for the drop zone
 * @param {Function} onFilesDropped - Callback when files are dropped, receives data URL
 * @param {Function} onError - Optional error handler
 * @returns {Object} { className, dragHandlers, error }
 */
export const useFileDropZone = (baseClassName, onFilesDropped, onError) => {
  const [className, setClassName] = useState(baseClassName);
  const [error, setError] = useState(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    setClassName(`${baseClassName} droppable`);
  }, [baseClassName]);

  const handleDragExit = useCallback(() => {
    setClassName(baseClassName);
  }, [baseClassName]);

  const handleDragLeave = useCallback(() => {
    setClassName(baseClassName);
  }, [baseClassName]);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    setClassName(baseClassName);
    setError(null);

    const files = Array.from(e.dataTransfer.files);
    try {
      const dataUrls = await readFilesAsDataURLs(files);
      dataUrls.forEach(url => onFilesDropped(url));
    } catch (err) {
      const errorMessage = `Failed to load files: ${err.message}`;
      setError(errorMessage);
      if (onError) {
        onError(errorMessage);
      }
    }
  }, [baseClassName, onFilesDropped, onError]);

  const dragHandlers = {
    onDragOver: handleDragOver,
    onDragExit: handleDragExit,
    onDragLeave: handleDragLeave,
    onDrop: handleDrop
  };

  return { className, dragHandlers, error };
};
