import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import ChecklistPage from './pages/ChecklistPage';
import PlanViewerPage from './pages/PlanViewerPage';
import { AppProvider } from './context/AppContext';

function App() {
    return (
        <AppProvider>
            <Router>
                <div className="min-h-screen flex flex-col">
                    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                            <div className="flex justify-between h-16">
                                <div className="flex">
                                    <div className="flex-shrink-0 flex items-center">
                                        <span className="text-xl font-bold text-indigo-600">SurveyPlan<span className="text-gray-900">AI</span></span>
                                    </div>
                                    <nav className="ml-6 flex space-x-8">
                                        <Link to="/" className="inline-flex items-center px-1 pt-1 border-b-2 border-indigo-500 text-sm font-medium text-gray-900">
                                            Upload Data
                                        </Link>
                                        <Link to="/checklist" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:text-gray-700 hover:border-gray-300">
                                            Checklist
                                        </Link>
                                        <Link to="/viewer" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:text-gray-700 hover:border-gray-300">
                                            Plan Viewer
                                        </Link>
                                    </nav>
                                </div>
                            </div>
                        </div>
                    </header>

                    <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
                        <Routes>
                            <Route path="/" element={<UploadPage />} />
                            <Route path="/checklist" element={<ChecklistPage />} />
                            <Route path="/viewer" element={<PlanViewerPage />} />
                        </Routes>
                    </main>
                </div>
            </Router>
        </AppProvider>
    );
}

export default App;
