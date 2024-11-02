import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, BarChart } from 'lucide-react';

const HeroPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      {/* Navigation */}
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <LineChart className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">FinanceAI</span>
            </div>
            <div className="flex items-center space-x-4">
              <button className="text-gray-600 hover:text-gray-900 px-3 py-2">About</button>
              <button className="text-gray-600 hover:text-gray-900 px-3 py-2">Features</button>
              <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">Get Started</button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 leading-tight">
              Transform Your Financial Reports Into
              <span className="text-blue-600"> Actionable Insights</span>
            </h1>
            <p className="mt-4 text-xl text-gray-600">
              Upload your financial reports and get instant AI-powered analysis, trends, and recommendations.
            </p>
            <div className="mt-8 space-x-4">
              <button className="bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-medium hover:bg-blue-700">Upload Reports</button>
              <button className="border border-gray-300 text-gray-700 px-6 py-3 rounded-lg text-lg font-medium hover:bg-gray-50">
                Learn More
              </button>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="relative"
          >
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Financial Overview</h3>
                  <p className="text-sm text-gray-500">Last 12 months</p>
                </div>
                <BarChart className="h-6 w-6 text-blue-600" />
              </div>
              <div className="space-y-4">
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div className="w-2/3 h-full bg-blue-600 rounded-full"></div>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div className="w-4/5 h-full bg-green-500 rounded-full"></div>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div className="w-1/2 h-full bg-yellow-500 rounded-full"></div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-gray-50 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900">Key Features</h2>
            <p className="mt-4 text-xl text-gray-600">Everything you need to analyze your financial reports</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: 'PDF Processing',
                description: 'Automatically extract and process data from your financial reports',
                icon: '📄'
              },
              {
                title: 'AI Analysis',
                description: 'Get intelligent insights and trends from your financial data',
                icon: '🤖'
              },
              {
                title: 'Interactive Dashboard',
                description: 'Visualize and explore your data with customizable charts',
                icon: '📊'
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="bg-white p-6 rounded-xl shadow-lg"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="bg-blue-600 rounded-2xl p-12 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">Ready to Transform Your Financial Analysis?</h2>
          <p className="text-xl text-blue-100 mb-8">Join thousands of companies using our platform to make better financial decisions.</p>
          <button className="bg-white text-blue-600 px-8 py-4 rounded-lg text-lg font-medium hover:bg-blue-50">Start Free Trial</button>
        </div>
      </div>
    </div>
  );
};

export default HeroPage;
