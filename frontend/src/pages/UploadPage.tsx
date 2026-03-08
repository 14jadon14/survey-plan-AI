import { useState, useRef, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { RefreshCw, FileImage, FileLineChart, FileCode, CheckCircle2, XCircle } from 'lucide-react';

export default function UploadPage() {
    const { state, setPlanData, setCadValidation, setCsvData, setPlanType } = useAppContext();
    const [loading, setLoading] = useState<string | null>(null);

    // Box 1 State
    const [anblsNumber, setAnblsNumber] = useState<string>('');

    // Box 2 State
    const [cadStatus, setCadStatus] = useState<'pending' | 'approved' | 'rejected'>('pending');
    const [csvStatus, setCsvStatus] = useState<'pending' | 'approved' | 'rejected'>('pending');
    const [csvCorners, setCsvCorners] = useState('');

    // Box 3 State
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
    const [rotation, setRotation] = useState(0);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Clean up object URLs to avoid memory leaks
    useEffect(() => {
        return () => {
            if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
        };
    }, []);

    // --- Action Handlers ---

    const handleGenerateAnbls = () => {
        setAnblsNumber('123456');
    };

    const handleDataUpload = async (endpoint: string, type: 'cad' | 'csv', event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setLoading(type);
        const setStatus = type === 'cad' ? setCadStatus : setCsvStatus;

        try {
            const formData = new FormData();
            formData.append('file', file);
            if (type === 'csv' && csvCorners) {
                formData.append('corners', csvCorners);
            }

            const res = await fetch(`http://localhost:8000${endpoint}`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();

            if (type === 'cad') {
                setCadValidation(data.validation, data.file_path);
                setStatus(data.valid ? 'approved' : 'rejected');
            } else {
                setCsvData(data.data);
                // Hardcoded to approved for now since we haven't implemented PNEZD check yet
                setStatus('approved');
            }
        } catch (err) {
            console.error(err);
            setStatus('rejected');
        } finally {
            setLoading(null);
        }
    };

    const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setImageFile(file);
            setRotation(0);

            if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
            setImagePreviewUrl(URL.createObjectURL(file));
        }
    };

    const handleRotate = () => {
        setRotation((prev) => (prev + 90) % 360);
    };

    const handleCrossVerify = async () => {
        if (!state.cadFilePath || !state.csvData || !csvCorners) return;

        setLoading('cross-verify');
        try {
            // 1. Identify rows in CSV that match the comma-separated corner IDs
            const cornerIds = csvCorners.split(',').map(s => s.trim());
            const cornersToVerify = state.csvData.filter((row: any) => {
                const pointId = String(row.point ?? row.Point ?? row[Object.keys(row)[0]]);
                return cornerIds.includes(pointId);
            }).map((row: any) => ({
                x: parseFloat(row.x ?? row.Easting ?? row.E ?? row[Object.keys(row)[2]]),
                y: parseFloat(row.y ?? row.Northing ?? row.N ?? row[Object.keys(row)[1]]),
                name: String(row.point ?? row.Point ?? row[Object.keys(row)[0]])
            }));

            if (cornersToVerify.length === 0) {
                alert("No matching points found in CSV for the provided corner IDs.");
                return;
            }

            // 2. Send to backend
            const res = await fetch(`http://localhost:8000/api/verify-cad-csv`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    cad_file_path: state.cadFilePath,
                    csv_corners: cornersToVerify
                }),
            });

            if (!res.ok) throw new Error(await res.text());
            const results = await res.json();

            if (results.valid) {
                alert(`SUCCESS: All ${results.matches_count} corner points matched CAD vertices!`);
            } else {
                alert(`FAILURE: ${results.missing_count} corner points were NOT found in the CAD file within 0.001m. Check points: ${results.missing_corners.map((p: any) => p.name).join(', ')}`);
            }
        } catch (err: any) {
            alert(`Failed cross-verification: ${err.message}`);
            console.error(err);
        } finally {
            setLoading(null);
        }
    };

    const handleSubmitPlan = async () => {
        if (!imageFile) return;

        setLoading('plan');
        try {
            const formData = new FormData();
            formData.append('file', imageFile);

            const res = await fetch(`http://localhost:8000/api/upload/plan`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();

            const validDetections = Array.isArray(data.detections) ? data.detections : [];
            // Preserve the rotated local URL if we ever add actual image rotation to the backend, 
            // for now just pass the rotated object URL to context
            setPlanData(imagePreviewUrl!, validDetections);

            alert('Plan successfully processed and submitted!');
        } catch (err: any) {
            alert(`Failed to process plan: ${err.message}`);
            console.error(err);
        } finally {
            setLoading(null);
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-6">

            {/* BOX 1: PLAN DATA */}
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h2 className="text-xl font-bold text-gray-900 mb-6 border-b pb-2">Plan Data</h2>
                <div className="flex flex-col sm:flex-row gap-6 justify-between items-start sm:items-end">

                    <div className="flex-1 w-full sm:max-w-xs">
                        <label className="block text-sm font-semibold text-gray-700 mb-2">Survey Plan Type</label>
                        <select
                            className="block w-full text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md border p-2 bg-gray-50"
                            value={state.planType}
                            onChange={(e) => setPlanType(e.target.value)}
                        >
                            <option value="Subdivision Plan">Subdivision Plan</option>
                            {/* Hidden until user asks for them: <option value="Condominium">Condominium</option> */}
                        </select>
                    </div>

                    <div className="flex-1 w-full sm:max-w-xs flex flex-col gap-2">
                        <label className="block text-sm font-semibold text-gray-700">ANBLS Number</label>
                        <div className="flex group">
                            <input
                                readOnly
                                value={anblsNumber}
                                placeholder="---"
                                className="block w-full border border-r-0 border-gray-300 rounded-l-md bg-gray-100 px-3 text-gray-600 focus:outline-none text-center font-mono"
                            />
                            <button
                                onClick={handleGenerateAnbls}
                                className="inline-flex justify-center items-center py-2 px-4 border border-indigo-600 shadow-sm text-sm font-medium rounded-r-md text-white bg-indigo-600 hover:bg-indigo-700 transition-colors whitespace-nowrap"
                            >
                                Generate ANBLS Number
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* BOX 2: DATA SUBMISSION */}
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h2 className="text-xl font-bold text-gray-900 mb-6 border-b pb-2">Data Submission</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

                    {/* CAD File */}
                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-5 flex flex-col items-center justify-between text-center min-h-[160px]">
                        {loading === 'cad' ? (
                            <div className="w-full h-full flex flex-col items-center justify-center text-indigo-600 space-y-3">
                                <RefreshCw className="animate-spin" size={32} />
                                <span className="font-medium animate-pulse">Running CAD Validation Layer Check...</span>
                            </div>
                        ) : (
                            <>
                                <div className="space-y-4 w-full">
                                    <div>
                                        <h3 className="text-sm font-bold text-slate-800 flex items-center justify-center gap-2">
                                            <FileCode size={18} className="text-indigo-500" /> CAD File (.dxf, .dwg)
                                        </h3>
                                        <p className="text-xs text-slate-500 mt-1">Check layer validation against NB requirements</p>
                                    </div>
                                    <input
                                        type="file" accept=".dxf,.dwg"
                                        onChange={(e) => handleDataUpload('/api/upload/cad', 'cad', e)}
                                        className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-indigo-100 file:text-indigo-700 hover:file:bg-indigo-200 transition-colors"
                                    />
                                </div>

                                {state.cadValidation && state.cadValidation.layers && state.cadValidation.topology && (
                                    <div className="mt-2 text-left text-[10px] space-y-1 bg-white p-2 rounded border border-slate-200">
                                        {state.cadValidation.message && (
                                            <p className="font-semibold text-slate-700 mb-1 border-b border-slate-100 pb-1">
                                                {state.cadValidation.message}
                                            </p>
                                        )}
                                        <p className={state.cadValidation.layers.valid ? "text-green-600" : "text-red-600"}>
                                            {state.cadValidation.layers.valid ? "✓ Layers OK" : "✗ Unauthorized Layers: " + (state.cadValidation.layers.unauthorized_layers?.slice(0, 3).join(', ') || 'None')}
                                        </p>
                                        <p className={state.cadValidation.topology.valid ? "text-green-600" : "text-red-600"}>
                                            {state.cadValidation.topology.valid ? "✓ Topology OK" : `✗ Gaps: ${state.cadValidation.topology.connectivity_gaps_count ?? 0} | Intersects: ${state.cadValidation.topology.intersections_count ?? 0}`}
                                        </p>
                                    </div>
                                )}

                                {/* Status Indicator */}
                                <div className="mt-4 pt-4 border-t border-slate-200 w-full flex justify-center">
                                    {cadStatus === 'pending' && <span className="text-xs font-semibold text-slate-400 capitalize tracking-wider">Awaiting Upload</span>}
                                    {cadStatus === 'approved' && <div className="flex items-center gap-1.5 text-green-600 font-bold bg-green-50 px-3 py-1 rounded-full text-sm border border-green-200"><CheckCircle2 size={16} /> VALIDATED</div>}
                                    {cadStatus === 'rejected' && <div className="flex items-center gap-1.5 text-red-600 font-bold bg-red-50 px-3 py-1 rounded-full text-sm border border-red-200"><XCircle size={16} /> FAILED</div>}
                                </div>
                            </>
                        )}
                    </div>

                    {/* CSV File */}
                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-5 flex flex-col items-center justify-between text-center min-h-[160px]">
                        {loading === 'csv' ? (
                            <div className="w-full h-full flex flex-col items-center justify-center text-indigo-600 space-y-3">
                                <RefreshCw className="animate-spin" size={32} />
                                <span className="font-medium animate-pulse">Running PNEZD Validation Check...</span>
                            </div>
                        ) : (
                            <>
                                <div className="space-y-4 w-full">
                                    <div>
                                        <h3 className="text-sm font-bold text-slate-800 flex items-center justify-center gap-2">
                                            <FileLineChart size={18} className="text-indigo-500" /> Coordinate Data (PNEZD)
                                        </h3>
                                        <p className="text-[10px] text-slate-500 mt-0.5">Supports .csv, .txt, .asc, .xyz, .pts</p>
                                    </div>

                                    {/* Subject Lot Corners Input */}
                                    <div className="text-left bg-white p-3 rounded border border-slate-200 shadow-sm">
                                        <label className="block text-xs font-semibold text-slate-700 mb-1">Subject Lot Corner Points</label>
                                        <input
                                            type="text"
                                            placeholder="e.g. 1,2,3,4,5"
                                            value={csvCorners}
                                            onChange={(e) => setCsvCorners(e.target.value)}
                                            className="block w-full text-xs p-1.5 border border-slate-300 rounded focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                                        />
                                        <p className="text-[10px] text-slate-400 mt-1 italic">Identify the PNEZD points mapping the boundary.</p>
                                    </div>

                                    <input
                                        type="file" accept=".csv,.txt,.asc,.xyz,.pts"
                                        onChange={(e) => handleDataUpload('/api/upload/csv', 'csv', e)}
                                        className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-indigo-100 file:text-indigo-700 hover:file:bg-indigo-200 transition-colors"
                                    />
                                </div>

                                {/* Status Indicator */}
                                <div className="mt-4 pt-4 border-t border-slate-200 w-full flex flex-col items-center gap-3">
                                    <div className="flex justify-center">
                                        {csvStatus === 'pending' && <span className="text-xs font-semibold text-slate-400 capitalize tracking-wider">Awaiting Upload</span>}
                                        {csvStatus === 'approved' && <div className="flex items-center gap-1.5 text-green-600 font-bold bg-green-50 px-3 py-1 rounded-full text-sm border border-green-200"><CheckCircle2 size={16} /> LOADED</div>}
                                        {csvStatus === 'rejected' && <div className="flex items-center gap-1.5 text-red-600 font-bold bg-red-50 px-3 py-1 rounded-full text-sm border border-red-200"><XCircle size={16} /> FAILED</div>}
                                    </div>

                                    {cadStatus === 'approved' && csvStatus === 'approved' && (
                                        <button
                                            onClick={handleCrossVerify}
                                            disabled={loading === 'cross-verify'}
                                            className="w-full py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-[10px] font-bold rounded flex items-center justify-center gap-1 shadow-sm transition-colors"
                                        >
                                            {loading === 'cross-verify' ? <RefreshCw className="animate-spin" size={12} /> : "CROSS-VERIFY WITH CAD"}
                                        </button>
                                    )}
                                </div>
                            </>
                        )}
                    </div>

                </div>
            </div>

            {/* BOX 3: PLAN SUBMISSION */}
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h2 className="text-xl font-bold text-gray-900 mb-6 border-b pb-2">Plan Submission</h2>

                <div className="flex flex-col md:flex-row gap-8 items-stretch min-h-[250px]">

                    {/* Left side: Upload */}
                    <div className="flex-1 flex flex-col justify-center items-center p-8 border-2 border-dashed border-slate-300 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors relative cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                        <FileImage size={48} className="text-slate-400 mb-4" />
                        <h3 className="text-lg font-bold text-slate-700">Upload Visual Survey Plan</h3>
                        <p className="text-sm text-slate-500 mt-2 max-w-xs text-center">Supported formats: JPG, PNG, TIF. Note: PDF files must be converted to image format first.</p>

                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".jpg,.jpeg,.png,.tif,.tiff"
                            onChange={handleImageSelect}
                            className="hidden"
                        />

                        {imageFile && (
                            <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 bg-indigo-100 text-indigo-800 rounded-full text-sm font-semibold border border-indigo-200">
                                <CheckCircle2 size={16} />
                                {imageFile.name}
                            </div>
                        )}
                    </div>

                    {/* Right side: Preview & action */}
                    <div className="w-full md:w-64 flex flex-col justify-between items-center space-y-4">

                        {/* Image Preview Window */}
                        <div className="w-full aspect-square bg-slate-800 rounded-lg overflow-hidden border-2 border-slate-200 relative shadow-inner flex items-center justify-center">
                            {imagePreviewUrl ? (
                                <img
                                    src={imagePreviewUrl}
                                    className="max-w-full max-h-full object-contain transition-transform duration-300 ease-in-out"
                                    style={{ transform: `rotate(${rotation}deg)` }}
                                    alt="Survey Plan Preview"
                                />
                            ) : (
                                <div className="text-slate-500 text-sm font-medium">No Image</div>
                            )}
                        </div>

                        {/* Rotate Action */}
                        <button
                            disabled={!imagePreviewUrl}
                            onClick={handleRotate}
                            className="w-full py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold rounded border border-slate-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                        >
                            Rotate <RefreshCw size={14} />
                        </button>

                        {/* Submit Action */}
                        <button
                            disabled={!imageFile || loading === 'plan'}
                            onClick={handleSubmitPlan}
                            className="w-full py-4 text-white font-bold bg-indigo-600 hover:bg-indigo-700 shadow-md rounded border border-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                        >
                            {loading === 'plan' ? (
                                <>
                                    <RefreshCw className="animate-spin mr-2" size={18} />
                                    Processing API...
                                </>
                            ) : 'Submit to AI Pipeline'}
                        </button>

                    </div>
                </div>

            </div>

        </div>
    );
}
