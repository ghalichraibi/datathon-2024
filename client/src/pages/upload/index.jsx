import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Upload, File, X, CheckCircle, AlertCircle, ArrowRight } from 'lucide-react';

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const navigate = useNavigate();

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    const pdfFiles = droppedFiles.filter(file => file.type === 'application/pdf');
    setFiles(prev => [...prev, ...pdfFiles]);
  }, []);

  const handleFileInput = useCallback((e) => {
    const selectedFiles = Array.from(e.target.files);
    const pdfFiles = selectedFiles.filter(file => file.type === 'application/pdf');
    setFiles(prev => [...prev, ...pdfFiles]);
  }, []);

  const removeFile = useCallback((indexToRemove) => {
    setFiles(prev => prev.filter((_, index) => index !== indexToRemove));
  }, []);

  const handleUpload = useCallback(async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    
    // Simuler le chargement et l'analyse
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    setIsUploading(false);
    setIsComplete(true);

    // Redirection après 3 secondes
    setTimeout(() => {
      navigate('/dashboard');
    }, 3000);
  }, [files, navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 via-white to-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Financial Reports</h2>
              
              {/* Upload Box */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
                  isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-500'
                }`}
              >
                <input
                  type="file"
                  accept=".pdf"
                  multiple
                  onChange={handleFileInput}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <Upload className="w-12 h-12 mx-auto mb-4 text-blue-500" />
                <p className="text-lg text-gray-600 mb-2">
                  Drag and drop your PDF files here
                </p>
                <p className="text-sm text-gray-500">
                  or click to browse your computer
                </p>
              </div>

              {/* Upload Button */}
              {files.length > 0 && !isUploading && !isComplete && (
                <motion.button
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  onClick={handleUpload}
                  className="mt-6 w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg shadow-blue-500/20"
                >
                  <span>Analyze Documents</span>
                  <ArrowRight className="w-4 h-4" />
                </motion.button>
              )}

              {/* Loading State */}
              {isUploading && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-6 text-center"
                >
                  <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                  <p className="text-lg font-medium text-gray-900">Analyzing your documents...</p>
                  <p className="text-sm text-gray-500">This might take a few moments</p>
                </motion.div>
              )}

              {/* Success Message */}
              {isComplete && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-6 text-center text-green-600"
                >
                  <CheckCircle className="w-16 h-16 mx-auto mb-4" />
                  <p className="text-lg font-medium">Analysis Complete!</p>
                  <p className="text-sm text-gray-500">Redirecting to dashboard...</p>
                </motion.div>
              )}
            </motion.div>
          </div>

          {/* Files List */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200"
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Uploaded Files</h3>
              
              <AnimatePresence mode="popLayout">
                {files.length === 0 ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center py-8 text-gray-500"
                  >
                    <File className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No files uploaded yet</p>
                  </motion.div>
                ) : (
                  <div className="space-y-2">
                    {files.map((file, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        className="flex items-center justify-between p-3 bg-gray-50 rounded-lg group"
                      >
                        <div className="flex items-center space-x-3">
                          <File className="w-5 h-5 text-blue-600" />
                          <span className="text-sm text-gray-600 truncate max-w-[150px]">
                            {file.name}
                          </span>
                        </div>
                        {!isUploading && !isComplete && (
                          <button
                            onClick={() => removeFile(index)}
                            className="opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <X className="w-4 h-4 text-gray-400 hover:text-red-500" />
                          </button>
                        )}
                      </motion.div>
                    ))}
                  </div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadPage;