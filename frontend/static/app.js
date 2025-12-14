// Minimal JS for demo. In this project most behavior is handled server-side.
// This file is extended to support dynamic state -> district selection.

console.log('App loaded');

// Simple state -> district mapping for India.
// NOTE: For a real production system you may want to load this from
// a separate JSON file or API. Here it is embedded for simplicity.
const indiaDistricts = {
  'Andhra Pradesh': [
    'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Krishna',
    'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam', 'Visakhapatnam',
    'Vizianagaram', 'West Godavari', 'YSR Kadapa'
  ],
  'Karnataka': [
    'Bagalkot', 'Ballari', 'Belagavi', 'Bengaluru Rural',
    'Bengaluru Urban', 'Bidar', 'Chamarajanagar', 'Chikkaballapur',
    'Chikkamagaluru', 'Chitradurga', 'Dakshina Kannada', 'Davangere',
    'Dharwad', 'Gadag', 'Hassan', 'Haveri', 'Kalaburagi', 'Kodagu',
    'Kolar', 'Koppal', 'Mandya', 'Mysuru', 'Raichur', 'Ramanagara',
    'Shivamogga', 'Tumakuru', 'Udupi', 'Uttara Kannada', 'Vijayapura',
    'Yadgir'
  ],
  'Jharkhand': [
    'Bokaro', 'Chatra', 'Deoghar', 'Dhanbad', 'Dumka', 'East Singhbhum',
    'Garhwa', 'Giridih', 'Godda', 'Gumla', 'Hazaribagh', 'Jamtara',
    'Khunti', 'Koderma', 'Latehar', 'Lohardaga', 'Pakur', 'Palamu',
    'Ramgarh', 'Ranchi', 'Sahibganj', 'Seraikela Kharsawan',
    'Simdega', 'West Singhbhum'
  ],
  'Maharashtra': [
    'Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed', 'Bhandara',
    'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli', 'Gondia',
    'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur', 'Latur', 'Mumbai City',
    'Mumbai Suburban', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik',
    'Osmanabad', 'Palghar', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri',
    'Sangli', 'Satara', 'Sindhudurg', 'Solapur', 'Thane', 'Wardha',
    'Washim', 'Yavatmal'
  ],
  'Tamil Nadu': [
    'Chennai', 'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul',
    'Erode', 'Kancheepuram', 'Kanniyakumari', 'Karur', 'Krishnagiri',
    'Madurai', 'Nagapattinam', 'Namakkal', 'Perambalur', 'Pudukkottai',
    'Ramanathapuram', 'Salem', 'Sivaganga', 'Thanjavur', 'The Nilgiris',
    'Theni', 'Thiruvallur', 'Thiruvarur', 'Thoothukudi',
    'Tiruchirappalli', 'Tirunelveli', 'Tiruppur', 'Tiruvannamalai',
    'Vellore', 'Viluppuram', 'Virudhunagar'
  ],
  'Uttar Pradesh': [
    'Agra', 'Aligarh', 'Allahabad', 'Ambedkar Nagar', 'Amethi',
    'Amroha', 'Auraiya', 'Azamgarh', 'Baghpat', 'Bahraich', 'Ballia',
    'Balrampur', 'Banda', 'Barabanki', 'Bareilly', 'Basti', 'Bhadohi',
    'Bijnor', 'Budaun', 'Bulandshahr', 'Chandauli', 'Chitrakoot',
    'Deoria', 'Etah', 'Etawah', 'Faizabad', 'Farrukhabad', 'Fatehpur',
    'Firozabad', 'Gautam Buddha Nagar', 'Ghaziabad', 'Ghazipur',
    'Gonda', 'Gorakhpur', 'Hamirpur', 'Hardoi', 'Hathras', 'Jalaun',
    'Jaunpur', 'Jhansi', 'Kannauj', 'Kanpur Dehat', 'Kanpur Nagar',
    'Kasganj', 'Kaushambi', 'Kheri', 'Kushinagar', 'Lalitpur', 'Lucknow',
    'Maharajganj', 'Mahoba', 'Mainpuri', 'Mathura', 'Mau', 'Meerut',
    'Mirzapur', 'Moradabad', 'Muzaffarnagar', 'Pilibhit', 'Pratapgarh',
    'Rae Bareli', 'Rampur', 'Saharanpur', 'Sambhal', 'Sant Kabir Nagar',
    'Shahjahanpur', 'Shamli', 'Shravasti', 'Siddharthnagar', 'Sitapur',
    'Sonbhadra', 'Sultanpur', 'Unnao', 'Varanasi'
  ],
  'Delhi': [
    'Central Delhi', 'East Delhi', 'New Delhi', 'North Delhi',
    'North East Delhi', 'North West Delhi', 'Shahdara', 'South Delhi',
    'South East Delhi', 'South West Delhi', 'West Delhi'
  ],
  // Add further states and districts here as needed.
};

function setupStateDistrictDropdowns() {
  const stateSelect = document.getElementById('state');
  const districtSelect = document.getElementById('district');
  // If district is not a <select> (e.g. manual text input), do nothing.
  if (!stateSelect || !districtSelect || districtSelect.tagName !== 'SELECT') return;

  stateSelect.addEventListener('change', () => {
    const state = stateSelect.value;
    const districts = indiaDistricts[state] || [];

    // Clear old options
    districtSelect.innerHTML = '';
    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = districts.length ? 'Select district' : 'No districts configured';
    districtSelect.appendChild(defaultOpt);

    districts.forEach((d) => {
      const opt = document.createElement('option');
      opt.value = d;
      opt.textContent = d;
      districtSelect.appendChild(opt);
    });
  });
}

function setupDoctorOwnVisitsFilter() {
  const table = document.getElementById('doctor-history-table');
  const checkbox = document.getElementById('only-my-visits');
  if (!table || !checkbox) return;

  const currentId = parseInt(table.getAttribute('data-current-doctor-id'), 10);
  if (!currentId) return;

  const rows = Array.from(table.querySelectorAll('tbody tr'));

  checkbox.addEventListener('change', () => {
    const onlyMine = checkbox.checked;
    rows.forEach((row) => {
      const rowDoctorId = parseInt(row.getAttribute('data-doctor-id'), 10);
      if (!onlyMine || rowDoctorId === currentId) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  });
}

function setupDoctorUseLastTreatment() {
  const btn = document.getElementById('use-last-treatment');
  const form = document.getElementById('add-record-form');
  if (!btn || !form) return;

  const diagnosisInput = form.querySelector('#diagnosis');
  const medicinesInput = form.querySelector('#medicines');
  const dosageInput = form.querySelector('#dosage');
  const statusSelect = form.querySelector('#treatment_status');
  const prescriptionText = form.querySelector('#prescription_text');

  btn.addEventListener('click', () => {
    if (diagnosisInput) diagnosisInput.value = btn.getAttribute('data-latest-diagnosis') || '';
    if (medicinesInput) medicinesInput.value = btn.getAttribute('data-latest-medicines') || '';
    if (dosageInput) dosageInput.value = btn.getAttribute('data-latest-dosage') || '';
    if (statusSelect) statusSelect.value = btn.getAttribute('data-latest-treatment-status') || '';
    if (prescriptionText) prescriptionText.value = btn.getAttribute('data-latest-prescription-text') || '';
  });
}

function setupHospitalVisitSearch() {
  const table = document.getElementById('hospital-visit-table');
  const input = document.getElementById('visit-search');
  if (!table || !input) return;

  const rows = Array.from(table.querySelectorAll('tbody tr'));

  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    rows.forEach((row) => {
      const text = (row.getAttribute('data-visit-search') || '').toLowerCase();
      row.style.display = !q || text.includes(q) ? '' : 'none';
    });
  });
}

function setupHospitalPatientSearch() {
  const table = document.getElementById('hospital-patient-table');
  const input = document.getElementById('patient-search');
  if (!table || !input) return;

  const rows = Array.from(table.querySelectorAll('tbody tr'));

  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    rows.forEach((row) => {
      const text = (row.getAttribute('data-patient-search') || '').toLowerCase();
      row.style.display = !q || text.includes(q) ? '' : 'none';
    });
  });
}

// Run immediately so the listener is attached as soon as script loads.
setupStateDistrictDropdowns();

// --- Doctor dashboard: client-side filter for medical history by status ---

function setupDoctorStatusFilter() {
  const table = document.getElementById('doctor-history-table');
  const filter = document.getElementById('status-filter');
  if (!table || !filter) return;

  const rows = Array.from(table.querySelectorAll('tbody tr'));

  filter.addEventListener('change', () => {
    const value = filter.value;
    rows.forEach((row) => {
      const status = (row.getAttribute('data-record-status') || '').trim();
      if (value === 'all' || status === value) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  });
}

// --- Hospital dashboard: emergency stats + doctor search filter ---

function setupHospitalEmergencyStats() {
  const target = document.getElementById('emergency-avg-text');
  if (!target) return;

  fetch('/analytics/ambulance')
    .then((res) => res.json())
    .then((data) => {
      const avg = data && typeof data.avg_response_time !== 'undefined'
        ? data.avg_response_time
        : null;
      if (avg !== null && avg !== undefined) {
        target.textContent = `Avg response: ${avg} min`;
      } else {
        target.textContent = 'Avg response: N/A';
      }
    })
    .catch(() => {
      target.textContent = 'Avg response: N/A';
    });
}

function setupHospitalDoctorSearch() {
  const table = document.getElementById('hospital-doctor-table');
  const input = document.getElementById('doctor-search');
  if (!table || !input) return;

  const rows = Array.from(table.querySelectorAll('tbody tr'));

  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    rows.forEach((row) => {
      const text = (row.getAttribute('data-doctor-search') || '').toLowerCase();
      row.style.display = !q || text.includes(q) ? '' : 'none';
    });
  });
}

function setupDoctorDeleteConfirm() {
  const forms = document.querySelectorAll('.doctor-delete-form');
  forms.forEach((form) => {
    form.addEventListener('submit', (e) => {
      const ok = window.confirm('Are you sure you want to delete this doctor? This cannot be undone.');
      if (!ok) {
        e.preventDefault();
      }
    });
  });
}

// Initialize dashboard helpers
setupHospitalVisitSearch();
setupHospitalPatientSearch();
setupDoctorStatusFilter();
setupDoctorOwnVisitsFilter();
setupDoctorUseLastTreatment();
setupHospitalEmergencyStats();
setupHospitalDoctorSearch();
setupDoctorDeleteConfirm();
