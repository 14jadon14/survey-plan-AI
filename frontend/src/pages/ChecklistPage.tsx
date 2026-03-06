import { useAppContext } from '../context/AppContext';
import ChecklistTree from '../components/ChecklistTree';

export default function ChecklistPage() {
    const { state } = useAppContext();

    return (
        <div className="bg-white shadow rounded-lg p-6 min-h-[500px]">
            <h2 className="text-xl font-bold text-gray-900 mb-6 border-b pb-2">Plan Checklist Audit - {state.planType}</h2>

            <ChecklistTree planType={state.planType} />
        </div>
    );
}
