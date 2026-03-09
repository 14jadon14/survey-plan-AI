import { useState } from 'react';
import { ChevronDown, ChevronRight, CheckCircle, XCircle } from 'lucide-react';

// ------------------------------------------------------------------
// Type definitions
// ------------------------------------------------------------------

type TriState = 'unanswered' | 'yes' | 'na';

interface CheckItem {
    id: string;
    text: string;
    status: TriState;
    notes: string;
    /** If set, this item only shows when the parent item (by id) is answered 'yes' */
    parentId?: string;
}

interface ChecklistCategory {
    id: string;
    title: string;
    items: CheckItem[];
}

// ------------------------------------------------------------------
// Real checklist data (New Brunswick Survey Plan Checklist)
// Items with a parentId are sub-questions that only appear when
// their parent item is answered "Yes".
// ------------------------------------------------------------------

const initialChecklistData: ChecklistCategory[] = [
    {
        id: 'system_admin',
        title: '1 \u2013 System and Administrative Information',
        items: [
            { id: 'sa1', text: 'Is the plan of survey in the registry system?', status: 'unanswered', notes: '' },
            { id: 'sa1a', text: 'Are current documents of title shown in the following format: Volume, page, document number, registration date? (G-4.19.12)', status: 'unanswered', notes: '', parentId: 'sa1' },
            { id: 'sa2', text: 'Is the plan of survey in the land titles system?', status: 'unanswered', notes: '' },
            { id: 'sa2a', text: 'Are current documents of title shown in the following format: transfer document number and date? (G-4.19.12)', status: 'unanswered', notes: '', parentId: 'sa2' },
            { id: 'sa3', text: 'Are file plan numbers and lot numbers of adjoining parcels shown, including names of owners and document information in the proper system formatting? (For registry: Volume, page, document number. For land titles: Transfer number.) (G-4.19.13)', status: 'unanswered', notes: '' },
            { id: 'sa4', text: 'Does the plan include property neighbouring, or on crown or forested land?', status: 'unanswered', notes: '' },
            { id: 'sa4a', text: 'If yes, a copy of the plan must be submitted to the Minister. (Crown Lands and Forests Act s. 11(1))', status: 'unanswered', notes: '', parentId: 'sa4' },
        ]
    },
    {
        id: 'general_plan',
        title: '2 \u2013 General Plan Requirements',
        items: [
            { id: 'gp1', text: 'Does the plan use text that is a minimum of 2mm in height? (G-4.23)', status: 'unanswered', notes: '' },
            { id: 'gp2', text: 'Is the plan drawn to one of the required scales (1:1000, 1:500, 1:2000, or 1:5000)? (CPA s. 84(2))', status: 'unanswered', notes: '' },
            { id: 'gp3', text: 'Is the plan on a permitted paper size (21.5cm \u00d7 35.5cm etc.)? (CPA s. 84(2))', status: 'unanswered', notes: '' },
            { id: 'gp4', text: 'Is adequate blank space left on the plan for ministerial/development officer approvals? (CPA s. 84(2))', status: 'unanswered', notes: '' },
            { id: 'gp5', text: 'Does the plan show the name of the owner and registration details (deed reference)? (CPA s. 84(3))', status: 'unanswered', notes: '' },
            { id: 'gp6', text: 'Are public streets labeled \u201cpublic street\u201d? (CPA s. 84(3))', status: 'unanswered', notes: '' },
            { id: 'gp7', text: 'Are future streets labeled \u201cfuture street\u201d? (CPA s. 84(3))', status: 'unanswered', notes: '' },
            { id: 'gp8', text: 'Is land for public purposes labelled as such? (CPA s. 84(3))', status: 'unanswered', notes: '' },
            { id: 'gp9', text: 'Does every lot have an identifying number or letter? (CPA s. 84(3))', status: 'unanswered', notes: '' },
            { id: 'gp10', text: 'Is the civic number (if available) shown? (CPA s. 84(3))', status: 'unanswered', notes: '' },
            { id: 'gp11', text: "Does the plan bear the Director of Surveys' approval? (Registry Act s. 50)", status: 'unanswered', notes: '' },
            { id: 'gp12', text: "Does the plan bear the Development Officer's certificate required prior to registration? (Registry Act, s. 46)", status: 'unanswered', notes: '' },
        ]
    },
    {
        id: 'title_block',
        title: '3 \u2013 Plan Title Block',
        items: [
            { id: 't1', text: 'Does the plan title include the type of plan? (G-4.14)', status: 'unanswered', notes: '' },
            { id: 't2', text: 'Does the plan title include the location, local government / parish / county? (G-4.14) (CPA s. 84(3a))', status: 'unanswered', notes: '' },
            { id: 't3', text: 'Does the plan contain proper scale and bar? (G-4.14)', status: 'unanswered', notes: '' },
            { id: 't4', text: "Does the plan contain the Land Surveyor's name, official number, and dated signature? (G-4.14)", status: 'unanswered', notes: '' },
            { id: 't5', text: 'Does the plan include a legend explaining all symbols used and all abbreviations used? (G-4.16)', status: 'unanswered', notes: '' },
            { id: 't6', text: 'Does the plan include a key plan? (G-4.20)', status: 'unanswered', notes: '' },
            { id: 't7', text: 'Does the plan include the street and subdivision accesses? (CPA s. 84(3)(a))', status: 'unanswered', notes: '' },
        ]
    },
    {
        id: 'main_body',
        title: '4 \u2013 Main Body of Plan',
        items: [
            { id: 'm1', text: 'Does the plan show all reasonable detail of parts of all adjacent properties? (G-4.3)', status: 'unanswered', notes: '' },
            { id: 'm2', text: 'Does the plan show, where possible, the position and form of all survey monuments and other evidence relevant to the survey? (G-4.4)', status: 'unanswered', notes: '' },
            // Natural boundary branch
            { id: 'm3', text: 'Does the plan show any natural boundaries? (G-4.5)', status: 'unanswered', notes: '' },
            { id: 'm3a', text: 'If this is a water boundary, does the plan use ordinary high-water mark to define the boundary? (G-4.5)', status: 'unanswered', notes: '', parentId: 'm3' },
            { id: 'm4', text: 'Does the plan include a north arrow properly referenced to a specific meridian, and not pointing to the lower edge of the plan? (G-4.6)', status: 'unanswered', notes: '' },
            { id: 'm5', text: 'Does the plan show notes including date of the survey and the coordinate system of the plan? (G-4.15)', status: 'unanswered', notes: '' },
            { id: 'm6', text: 'Does each line of the plan pertaining to the primary lot have an associated azimuth? (G-4.8)', status: 'unanswered', notes: '' },
            { id: 'm7', text: 'Are bearings expressed in grid azimuth? (Survey Act s. 4(1))', status: 'unanswered', notes: '' },
            { id: 'm8', text: 'Does each line of the plan pertaining to the primary lot have an associated distance reduced to New Brunswick Double Stereographic expressed in meters? (G-4.11) (Survey Act s. 4(1))', status: 'unanswered', notes: '' },
            { id: 'm9', text: 'Is a solid black line used to indicate all boundaries determined in the survey, with heavier line weight for exterior boundaries of parcels being dealt with? (G-4.19.1)', status: 'unanswered', notes: '' },
            { id: 'm10', text: 'Does the plan show the nature and position of all monuments marking boundaries, and all permanent structures to which they have been referenced? (G-4.19.2)', status: 'unanswered', notes: '' },
            { id: 'm11', text: 'Where a monument is placed on an existing boundary line, does the plan show evidence of the line on either side of the monument? (G-4.19.2)', status: 'unanswered', notes: '' },
            { id: 'm12', text: 'Does the plan show the nature and position of monuments marking any line from which azimuths have been deduced and any monuments to which a surveyed connection has been made? (G-4.19.3)', status: 'unanswered', notes: '' },
            { id: 'm13', text: 'If the surveyor has not monumented a new or existing boundary because of the survey, does the plan contain a note explaining the absence of monumentation? (G-4.22)', status: 'unanswered', notes: '' },
            // Curve information branch
            { id: 'm14', text: 'Does the plan include any curve information?', status: 'unanswered', notes: '' },
            { id: 'm14a', text: 'Does the plan show arc lengths and radii of circular curves, and arc lengths between adjacent monuments marking a curved boundary? (G-4.19.6)', status: 'unanswered', notes: '', parentId: 'm14' },
            { id: 'm14b', text: 'Are chord lengths and azimuths between adjacent monuments on curves shown, including coordinates of radius points when applicable? (G-4.19.6)', status: 'unanswered', notes: '', parentId: 'm14' },
            { id: 'm14c', text: 'Where spirals are present, does the plan show length and terminal radius of the spiral and lengths and azimuths of chords between monuments on the spiral? (G-4.19.7)', status: 'unanswered', notes: '', parentId: 'm14' },
            { id: 'm14d', text: 'Are central angles shown for all curves? (CPA s. 84(4))', status: 'unanswered', notes: '', parentId: 'm14' },
            { id: 'm15', text: 'Does the plan show designations of each lot, road, street, etc., and widths of each new road, street, etc.? (G-4.19.8)', status: 'unanswered', notes: '' },
            { id: 'm16', text: 'Does the plan show designation of all existing lots, parcels, roads, and streets adjacent to the survey, and widths of adjacent roadways, plus record numbers and locations where records are available? (G-4.19.9)', status: 'unanswered', notes: '' },
            { id: 'm17', text: 'Is the area of each new lot shown within the appropriate lot, or alternatively in tabular form, using the recommended precision? (G-4.19.10)', status: 'unanswered', notes: '' },
            { id: 'm18', text: 'Is the Service New Brunswick Parcel Identifier (PID) shown where available? (G-4.19.14)', status: 'unanswered', notes: '' },
            // Easement branch
            { id: 'm19', text: 'Does the plan include any easements?', status: 'unanswered', notes: '' },
            { id: 'm19a', text: 'Are all easements or rights-of-way described in title documents shown on the plan? (G-4.19.15)', status: 'unanswered', notes: '', parentId: 'm19' },
            { id: 'm19b', text: 'Are easements labelled with their purpose (e.g., \u201cDrainage Easement\u201d)? (CPA s. 84(3))', status: 'unanswered', notes: '', parentId: 'm19' },
            { id: 'm19c', text: 'Does the plan show all easements and rights-of-way affecting the land, including any that may have been acquired by long-term use (20 to 40 years)? (RSNB 2011, c.143)', status: 'unanswered', notes: '', parentId: 'm19' },
            { id: 'm20', text: 'Are signatures handwritten (not printed), showing the name or signing authority clearly? (G-4.21)', status: 'unanswered', notes: '' },
            // Integrated Survey Area branch
            { id: 'm21', text: 'Is the plan within an Integrated Survey Area?', status: 'unanswered', notes: '' },
            { id: 'm21a', text: 'Are the ties to coordinate monuments shown? (Surveys Act s. 7)', status: 'unanswered', notes: '', parentId: 'm21' },
        ]
    },
    {
        id: 'electronic',
        title: '5 \u2013 Electronic Plan Submission',
        items: [
            { id: 'e1', text: 'Is the digital scan authenticated by the surveyor as an exact copy of the original? (Registry Act, s. 19) (Land Titles Act, s. 17.2(5))', status: 'unanswered', notes: '' },
            { id: 'e2', text: 'Is the digital image compliant with registry scanning requirements? (Registry Act, s. 19)', status: 'unanswered', notes: '' },
        ]
    },
    {
        id: 'condos',
        title: '6 \u2013 Condominiums',
        items: [
            { id: 'c1', text: 'Does the plan show the perimeter of land and buildings as required? (Cond. Act s. 7(1))', status: 'unanswered', notes: '' },
            { id: 'c2', text: 'Are unit boundaries fully described?', status: 'unanswered', notes: '' },
            { id: 'c3', text: "Is the Surveyor's Certificate included stating buildings are constructed and diagrams are accurate? (Cond. Act s. 7(1)(d))", status: 'unanswered', notes: '' },
            { id: 'c4', text: 'For bare-land condos, are unit boundaries tied to coordinate monuments? (Cond. Act s. 7(2))', status: 'unanswered', notes: '' },
            { id: 'c5', text: 'Does the plan bear subdivision certification under the Community Planning Act if applicable? (Cond. Act s. 8(1))', status: 'unanswered', notes: '' },
            { id: 'c6', text: 'Does the plan show the shape and dimensions of all unit perimeters and building outlines as required? (Condo Act s. 7(1))', status: 'unanswered', notes: '' },
        ]
    },
    {
        id: 'highways',
        title: '7 \u2013 Highways',
        items: [
            { id: 'h1', text: 'Is any portion of the land adjacent to a provincially designated highway (arterial, collector, or local)? (Highway Act, s. 1)', status: 'unanswered', notes: '' },
            { id: 'h1a', text: 'If new roads are shown, are they clearly labeled as public, private, or \u201cfuture streets\u201d consistent with CPA and Highway Act definitions?', status: 'unanswered', notes: '', parentId: 'h1' },
            { id: 'h1b', text: 'If a road is intended to become a provincial highway, has the plan been prepared to provincial standards for acceptance?', status: 'unanswered', notes: '', parentId: 'h1' },
            { id: 'h1c', text: 'Are the boundaries of any adjacent highway shown accurately and completely?', status: 'unanswered', notes: '', parentId: 'h1' },
            { id: 'h1d', text: 'Is any highway widening lines or historical right-of-way limits shown if applicable?', status: 'unanswered', notes: '', parentId: 'h1' },
            { id: 'h1e', text: 'Does the plan show (or require) new access to a provincial highway? If yes, is a Highway Usage Permit required or referenced? (s. 44.1)', status: 'unanswered', notes: '', parentId: 'h1' },
            { id: 'h1f', text: 'Are any encroachments into the highway right-of-way shown and identified?', status: 'unanswered', notes: '', parentId: 'h1' },
            { id: 'h1g', text: 'Is the land located within a designated \u201cdevelopment area\u201d under the Highway Act, and has this been accounted for in the plan?', status: 'unanswered', notes: '', parentId: 'h1' },
        ]
    }
];

// ------------------------------------------------------------------
// Component
// ------------------------------------------------------------------

export default function ChecklistTree({ planType = 'Standard' }: { planType?: string }) {
    const [data, setData] = useState<ChecklistCategory[]>(initialChecklistData);
    const [expanded, setExpanded] = useState<Record<string, boolean>>({});
    const [sectionNotes, setSectionNotes] = useState<Record<string, string>>({});

    const toggleExpand = (id: string) => {
        setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
    };

    const updateItemStatus = (categoryId: string, itemId: string, newStatus: TriState) => {
        setData(prev => prev.map(cat => {
            if (cat.id !== categoryId) return cat;
            return {
                ...cat,
                items: cat.items.map(item =>
                    item.id === itemId ? { ...item, status: newStatus } : item
                )
            };
        }));
    };

    /** Returns true if the parent item for this item is answered 'yes' (or item has no parent) */
    const isVisible = (item: CheckItem, allItems: CheckItem[]): boolean => {
        if (!item.parentId) return true;
        const parent = allItems.find(i => i.id === item.parentId);
        if (!parent) return true;
        return parent.status === 'yes';
    };

    const renderItem = (categoryId: string, item: CheckItem, isChild: boolean) => {
        return (
            <div
                key={item.id}
                className={`mb-1 p-2 border rounded-md ${isChild ? 'ml-6 bg-blue-50 border-blue-100' : 'ml-0 bg-gray-50 border-gray-200'}`}
            >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <span className="text-sm font-medium text-gray-800 flex-1">{item.text}</span>
                    <div className="flex items-center space-x-2 shrink-0">
                        <button
                            onClick={() => updateItemStatus(categoryId, item.id, 'yes')}
                            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-bold ${item.status === 'yes' ? 'bg-green-100 text-green-800 border border-green-300' : 'bg-white border text-gray-400 hover:bg-gray-50'}`}
                        >
                            <CheckCircle size={14} />
                            <span>YES</span>
                        </button>
                        <button
                            onClick={() => updateItemStatus(categoryId, item.id, 'na')}
                            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-bold ${item.status === 'na' ? 'bg-blue-100 text-blue-800 border border-blue-300' : 'bg-white border text-gray-400 hover:bg-gray-50'}`}
                        >
                            <XCircle size={14} />
                            <span>N/A</span>
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    const renderCategory = (category: ChecklistCategory) => {
        // Hide Condominiums unless plan type is Condominium
        if (category.id === 'condos' && planType !== 'Condominium') return null;

        const isExpanded = expanded[category.id] ?? true;

        // Only count visible items for progress
        const visibleItems = category.items.filter(item => isVisible(item, category.items));
        const answeredItems = visibleItems.filter(i => i.status !== 'unanswered').length;
        const totalItems = visibleItems.length;
        const isComplete = answeredItems === totalItems && totalItems > 0;

        return (
            <div key={category.id} className="mb-2 rounded-lg border border-gray-200 shadow-sm overflow-hidden">
                <div
                    className={`flex items-center justify-between px-2 py-1.5 cursor-pointer ${isExpanded ? 'bg-sky-50' : 'bg-white hover:bg-sky-50 transition-colors'}`}
                    onClick={() => toggleExpand(category.id)}
                >
                    <div className="flex items-center space-x-2">
                        {isExpanded
                            ? <ChevronDown size={18} className="text-sky-500" />
                            : <ChevronRight size={18} className="text-sky-500" />}
                        <span className="font-semibold text-gray-900">{category.title}</span>
                    </div>
                    <div className="flex items-center space-x-3">
                        <span className="text-xs text-gray-500">{answeredItems} / {totalItems}</span>
                        {isComplete && <CheckCircle size={16} className="text-green-500" />}
                    </div>
                </div>

                {isExpanded && (
                    <div className="px-2 py-1.5 bg-white space-y-1">
                        {category.items.map(item => {
                            const visible = isVisible(item, category.items);
                            if (!visible) return null;
                            return renderItem(category.id, item, !!item.parentId);
                        })}

                        {/* Section-specific notes */}
                        <div className="mt-2 pt-2 border-t border-slate-100">
                            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Section Comments</label>
                            <textarea
                                className="w-full text-sm p-3 border border-slate-200 rounded-md focus:ring-sky-500 focus:border-sky-500 bg-slate-50"
                                rows={2}
                                value={sectionNotes[category.id] || ''}
                                onChange={(e) => setSectionNotes(prev => ({ ...prev, [category.id]: e.target.value }))}
                                placeholder={`Comments for ${category.title}...`}
                            />
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="w-full">
            <div className="space-y-4">
                {data.map(renderCategory)}
            </div>

            <div className="mt-6 flex justify-center border-t border-slate-100 pt-4">
                <button
                    onClick={() => alert("Checklist Submitted! (Database logging to be implemented)")}
                    className="w-full max-w-md py-3 bg-sky-600 hover:bg-sky-700 text-white text-lg font-bold rounded-lg shadow-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] mb-4"
                >
                    Submit Checklist
                </button>
            </div>
        </div>
    );
}
