import { useAppContext } from '../context/AppContext';
import ChecklistTree from '../components/ChecklistTree';

export default function ChecklistPage() {
    const { state } = useAppContext();

    return (
        <div className="max-w-5xl mx-auto bg-white shadow rounded-lg pt-2 px-4 pb-4 min-h-[500px]">
            <h2 className="text-xl font-bold text-gray-900 mb-2 border-b pb-1">
                Instrument #{state.anblsNumber || '([ANBLS])'} - Checklist
            </h2>

            <ChecklistTree planType={state.planType} />
        </div>
    );
}
