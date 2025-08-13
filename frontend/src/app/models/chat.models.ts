// src/app/models/chat.models.ts - UPDATED
export enum ResponseType {
  TABLE = 'table',
  TREE = 'tree',
  CHART = 'chart',
  STATUS = 'status',
  FORM = 'form',
  GENERIC = 'generic',
  ERROR = 'error'
}

export enum InteractionType {
  APPROVE = 'approve',
  REJECT = 'reject',
  MODIFY = 'modify',
  CANCEL = 'cancel'
}

export interface UIRecommendation {
  display_type: ResponseType;
  columns?: string[];
  sortable_columns?: string[];
  filterable_columns?: string[];
  grouping_options?: string[];
  actions: string[];
  export_formats: string[];
}

export interface HumanApprovalRequest {
  id: string;
  step: string;
  message: string;
  data: any;
  auto_approve: boolean;
  timeout_seconds: number;
  created_at: string;
}

// UPDATED: ChatMessage now stores full response
export interface ChatMessage {
  id: string;
  type: 'user' | 'agent' | 'approval';
  content: string;
  timestamp: string;
  session_id: string;
  response: AgentResponse | null; // Changed from optional to null union type
}

// UPDATED: AgentResponse to match API response structure
export interface AgentResponse {
  id: string;
  success: boolean;
  type: ResponseType; // Changed from response_type to type
  data?: any;
  error?: string;
  message?: string;
  suggestions?: string[]; // Add suggestions from API
  ui_recommendations?: UIRecommendation;
  next_actions?: string[];
  approval_request?: HumanApprovalRequest;
  session_id: string;
  timestamp: string;
}

export interface ComponentData {
  id: string;
  name: string;
  part_number?: string;
  manufacturer?: string;
  description?: string;
  value?: string;
  quantity: number;
  designator?: string;
  confidence: number;
  metadata: Record<string, any>;
}

export interface BOMData {
  id: string;
  name: string;
  project?: string;
  components: ComponentData[];
  created_at: string;
  modified_at: string;
  metadata: Record<string, any>;
  component_count: number;
}