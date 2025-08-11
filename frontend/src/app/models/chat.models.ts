// src/app/models/chat.models.ts
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

export interface ChatMessage {
  id: string;
  type: 'user' | 'agent' | 'approval';
  content: string;
  timestamp: string;
  session_id: string;
}

export interface AgentResponse {
  id: string;
  success: boolean;
  response_type: ResponseType;
  data?: any;
  error?: string;
  message?: string;
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