import { createContext, useContext, useState, ReactNode } from 'react';

interface AppState {
    planImageUrl: string | null;
    detections: any[];
    cadFilePath: string | null;
    cadValidation: any | null;
    csvData: any | null;
    subjectLot: any | null;
    planType: string;
}

interface AppContextType {
    state: AppState;
    setPlanData: (imageUrl: string, detections: any[]) => void;
    setCadValidation: (data: any, cadFilePath?: string | null) => void;
    setCsvData: (data: any) => void;
    setSubjectLot: (data: any) => void;
    setPlanType: (type: string) => void;
}

const defaultState: AppState = {
    planImageUrl: null,
    detections: [],
    cadFilePath: null,
    cadValidation: null,
    csvData: null,
    subjectLot: null,
    planType: 'Subdivision Plan'
};

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
    const [state, setState] = useState<AppState>(defaultState);

    const setPlanData = (planImageUrl: string, detections: any[]) => {
        setState(prev => ({ ...prev, planImageUrl, detections }));
    };

    const setCadValidation = (cadValidation: any, cadFilePath: string | null = null) => {
        setState(prev => ({ ...prev, cadValidation, cadFilePath: cadFilePath || prev.cadFilePath }));
    };

    const setCsvData = (csvData: any) => {
        setState(prev => ({ ...prev, csvData }));
    };

    const setSubjectLot = (subjectLot: any) => {
        setState(prev => ({ ...prev, subjectLot }));
    };

    const setPlanType = (planType: string) => {
        setState(prev => ({ ...prev, planType }));
    };

    return (
        <AppContext.Provider value={{ state, setPlanData, setCadValidation, setCsvData, setSubjectLot, setPlanType }}>
            {children}
        </AppContext.Provider>
    );
}

export function useAppContext() {
    const context = useContext(AppContext);
    if (context === undefined) {
        throw new Error('useAppContext must be used within an AppProvider');
    }
    return context;
}
