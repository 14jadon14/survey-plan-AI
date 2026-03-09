import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import ChecklistPage from './pages/ChecklistPage';
import PlanViewerPage from './pages/PlanViewerPage';
import { AppProvider } from './context/AppContext';
import eefLogo from './assets/eef_logo.png';
import ggeLogo from './assets/gge_logo.png';
import unbLogo from './assets/unb_symposium_logo.png';

function App() {
    return (
        <AppProvider>
            <Router>
                <div className="min-h-screen flex flex-col bg-gray-50">
                    <header className="bg-white border-b border-sky-100 sticky top-0 z-50 shadow-sm">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                            <div className="flex justify-between h-14">
                                {/* Left side: Branding & Nav */}
                                <div className="flex items-center space-x-8">
                                    <div className="flex items-center space-x-4">
                                        <img src={ggeLogo} alt="GGE Logo" className="h-12 w-auto" />
                                        <div className="h-8 w-px bg-gray-200 mx-2"></div>
                                        <span className="text-xl font-bold text-sky-600">SurveyPlan<span className="text-gray-900">AI</span></span>
                                    </div>
                                    <nav className="flex space-x-8">
                                        <Link to="/" className="inline-flex items-center px-1 pt-1 border-b-2 border-sky-500 text-sm font-medium text-gray-900">
                                            Upload Data
                                        </Link>
                                        <Link to="/checklist" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:text-sky-600 hover:border-sky-300">
                                            Checklist
                                        </Link>
                                        <Link to="/viewer" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:text-sky-600 hover:border-sky-300">
                                            Plan Viewer
                                        </Link>
                                    </nav>
                                </div>
                                {/* Right side: Branding partner logos */}
                                <div className="flex items-center space-x-6">
                                    <img src={eefLogo} alt="EEF Logo" className="h-10 w-auto" />
                                    <img src={unbLogo} alt="UNB Logo" className="h-10 w-auto" />
                                </div>
                            </div>
                        </div>
                    </header>

                    <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-2">
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
