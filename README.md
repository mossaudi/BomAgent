# BOM Agent Chat Application

Modern Angular chat interface for the Intelligent BOM Management Agent with human-in-the-loop functionality.

## Features

- 🤖 **Real-time Chat Interface**: Interactive conversation with the BOM agent
- 🔄 **Human-in-the-Loop**: Approval workflow for critical operations
- 📊 **Smart Data Visualization**: Automatic rendering based on response type
  - Tables for component data with sorting and filtering
  - Tree view for hierarchical BOM structures
  - Status cards for memory and session information
  - JSON viewer for raw data inspection
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🌓 **Dark Mode Support**: Automatic dark mode based on system preferences
- ⚡ **Real-time Updates**: Live session management and memory tracking
- 🎨 **Modern UI/UX**: Clean, professional interface with smooth animations

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+ (for backend)
- Angular CLI 17+

### Installation

1. **Frontend Setup:**
```bash
npm install -g @angular/cli
npm install
ng serve
```

2. **Backend Setup:**
```bash
pip install -r requirements.txt
python server.py
```

3. **Environment Configuration:**
```bash
# Create .env file
GOOGLE_API_KEY=your_gemini_api_key
SILICON_EXPERT_USERNAME=your_username
SILICON_EXPERT_API_KEY=your_api_key
```

### Usage

1. Open http://localhost:4200 in your browser
2. Start chatting with the agent:
   - "Analyze schematic at [URL]"
   - "Show me memory status"
   - "Create BOM from components"
   - "List existing BOMs"

### Architecture

- **Frontend**: Angular 17 with TypeScript
- **Backend**: FastAPI with async/await
- **Agent**: LangGraph with human-in-the-loop
- **UI Components**: Custom responsive components
- **State Management**: RxJS observables
- **Styling**: SCSS with CSS Grid/Flexbox

### Response Types

The agent returns structured responses that automatically render as:

- **Table**: Component lists with sorting, filtering, and actions
- **Tree**: Hierarchical BOM structures with expand/collapse
- **Status**: Session and memory information cards
- **Form**: Approval requests with interactive buttons
- **JSON**: Raw data with syntax highlighting

### Human Approval Workflow

1. Agent processes request and identifies approval points
2. UI displays approval panel with context
3. User can approve, reject, or modify with feedback
4. Agent continues workflow based on human decision
5. Results displayed with recommended next actions

## Development

### Project Structure

```
src/
├── app/
│   ├── components/          # UI components
│   │   ├── chat-interface/  # Main chat interface
│   │   ├── table-renderer/  # Data table component
│   │   ├── tree-renderer/   # Tree view component
│   │   └── approval-panel/  # Human approval UI
│   ├── models/             # TypeScript interfaces
│   ├── services/           # HTTP and business logic
│   └── styles/            # SCSS stylesheets
└── assets/                # Static assets
```

### Customization

- **Themes**: Modify SCSS variables in `styles.scss`
- **Components**: Extend base renderer components
- **Response Types**: Add new response types and renderers
- **API**: Extend ChatService for additional endpoints

### Building for Production

```bash
ng build --prod
```

The built application will be in `dist/` directory, ready for deployment.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.