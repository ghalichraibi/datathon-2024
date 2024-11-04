import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, FileText, Brain, ArrowRight, Upload } from 'lucide-react';
import LogoSection from 'components/logo';

const HeroPage = () => {
  const [activeFeature, setActiveFeature] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setIsVisible(false);
      setTimeout(() => {
        setActiveFeature((prev) => (prev + 1) % 3);
        setIsVisible(true);
      }, 200);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      title: 'Financial Analysis',
      description: 'Real-time metrics & KPIs',
      icon: <TrendingUp className="w-6 h-6" />,
      color: 'from-blue-600 to-indigo-700'
    },
    {
      title: 'PDF Processing',
      description: 'Automated data extraction',
      icon: <FileText className="w-6 h-6" />,
      color: 'from-violet-600 to-purple-700'
    },
    {
      title: 'AI Insights',
      description: 'Predictive analytics',
      icon: <Brain className="w-6 h-6" />,
      color: 'from-blue-700 to-indigo-800'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 via-white to-gray-50 overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute w-full h-full bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-100/40 via-white to-white"></div>
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.1, 0.2, 0.1]
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            repeatType: 'reverse'
          }}
          className="absolute top-0 right-0 w-1/2 h-1/2 bg-blue-200/20 rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            opacity: [0.1, 0.15, 0.1]
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            repeatType: 'reverse'
          }}
          className="absolute bottom-0 left-0 w-1/2 h-1/2 bg-indigo-200/20 rounded-full blur-3xl"
        />
      </div>

      {/* Navigation */}
      <nav className="relative z-10 bg-white/80 backdrop-blur-xl border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-20">
            <div className="flex items-center">
              <LogoSection to="/" />
            </div>
            <div className="flex items-center space-x-8">
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="hidden md:flex space-x-8"
              >
                <a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors">
                  Features
                </a>
                <a href="#about" className="text-gray-600 hover:text-blue-600 transition-colors">
                  About
                </a>
                <a href="#pricing" className="text-gray-600 hover:text-blue-600 transition-colors">
                  Pricing
                </a>
              </motion.div>
              <motion.button
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-2.5 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 flex items-center space-x-2 group shadow-lg shadow-blue-500/20 cursor-pointer"
                onClick={() => (window.location.href = 'upload')}
              >
                <span>Get Started</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </motion.button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-32">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Left Column */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="space-y-8">
              <h1 className="text-5xl lg:text-6xl font-bold leading-tight text-gray-900">
                Transform Financial Reports Into
                <span className="block mt-2 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Strategic Intelligence
                </span>
              </h1>
              <p className="text-xl text-gray-600 leading-relaxed">
                Enterprise-grade AI that transforms complex financial documents into actionable intelligence. Trusted by industry leaders
                for strategic decision-making.
              </p>
              <div className="flex flex-wrap gap-4">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-lg text-lg font-medium hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 flex items-center space-x-2 shadow-lg shadow-blue-500/20"
                  onClick={() => (window.location.href = 'upload')}
                >
                  <Upload className="w-5 h-5" />
                  <span className="cursor-pointer">Upload Reports</span>
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="bg-white text-gray-800 px-8 py-4 rounded-lg text-lg font-medium hover:bg-gray-50 transition-all duration-200 border border-gray-200 shadow-lg shadow-gray-200/20"
                >
                  Watch Demo
                </motion.button>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-8 pt-8 border-t border-gray-200">
                {[
                  { value: '99%', label: 'Accuracy' },
                  { value: '75%', label: 'Time Saved' },
                  { value: '500+', label: 'Companies' }
                ].map((stat, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                  >
                    <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
                    <div className="text-sm text-gray-500">{stat.label}</div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Right Column - Interactive Feature Display */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-100/50 to-indigo-100/50 rounded-2xl blur-3xl"></div>
              <div className="relative bg-white/70 backdrop-blur-xl rounded-2xl border border-gray-200 p-8 shadow-xl">
                <AnimatePresence mode="wait">
                  {isVisible && (
                    <motion.div
                      key={activeFeature}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.2 }}
                      className="space-y-6"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className={`p-3 rounded-lg bg-gradient-to-r ${features[activeFeature].color} text-white`}>
                            {features[activeFeature].icon}
                          </div>
                          <div>
                            <h3 className="text-lg font-semibold text-gray-900">{features[activeFeature].title}</h3>
                            <p className="text-sm text-gray-500">{features[activeFeature].description}</p>
                          </div>
                        </div>
                      </div>

                      <div className="space-y-4">
                        {[...Array(3)].map((_, i) => (
                          <div key={i} className="space-y-2">
                            <div className="flex justify-between text-sm text-gray-500">
                              <span>Metric {i + 1}</span>
                              <span>{Math.floor(Math.random() * 100)}%</span>
                            </div>
                            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.random() * 100}%` }}
                                transition={{ duration: 1, ease: 'easeOut' }}
                                className={`h-full rounded-full bg-gradient-to-r ${features[activeFeature].color}`}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Companies Ticker */}
        <div className="relative py-12 bg-gradient-to-r from-gray-50 to-white border-t border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-center space-x-12">
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-gray-500 font-medium">
                Trusted by industry leaders
              </motion.p>
              {['Hydro One', 'Empire', 'CN', 'Bell', 'Couche-Tard'].map((company, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: index * 0.1 }}
                  className="text-gray-600 font-semibold"
                >
                  {company}
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeroPage;
