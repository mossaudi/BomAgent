// src/app/services/chat.service.ts - UPDATED
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, Subject } from 'rxjs';
import { ChatMessage, AgentResponse, HumanApprovalRequest } from '../models/chat.models';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private readonly apiUrl = environment.apiUrl;
  private messagesSubject = new BehaviorSubject<ChatMessage[]>([]);
  private pendingApprovalSubject = new Subject<HumanApprovalRequest>();

  public messages$ = this.messagesSubject.asObservable();
  public pendingApproval$ = this.pendingApprovalSubject.asObservable();

  private sessionId: string = this.generateSessionId();
  private messageHistory: ChatMessage[] = [];

  constructor(private http: HttpClient) {}

  private generateSessionId(): string {
    return 'session-' + Math.random().toString(36).substr(2, 9);
  }

  getSessionId(): string {
    return this.sessionId;
  }

  async sendMessage(content: string): Promise<void> {
    // Add user message
    const userMessage: ChatMessage = {
      id: this.generateId(),
      type: 'user',
      content,
      timestamp: new Date().toISOString(),
      session_id: this.sessionId,
      response: null // User messages don't have responses
    };

    this.addMessage(userMessage);

    try {
      const response = await this.http.post<AgentResponse>(`${this.apiUrl}/chat`, {
        message: content,
        session_id: this.sessionId
      }).toPromise();

      if (response) {
        this.handleAgentResponse(response);
      }
    } catch (error) {
      this.handleError(error);
    }
  }

  // UPDATED: Store full response in message
  private handleAgentResponse(response: AgentResponse): void {
    console.log('Handling agent response:', response); // Debug log

    // Check if approval is required
    if (response.approval_request) {
      this.pendingApprovalSubject.next(response.approval_request);

      const approvalMessage: ChatMessage = {
        id: response.id,
        type: 'approval',
        content: response.approval_request.message,
        timestamp: response.timestamp,
        session_id: this.sessionId,
        response: response // Store full response
      };

      this.addMessage(approvalMessage);
    } else {
      // Regular agent response - STORE THE FULL RESPONSE
      const agentMessage: ChatMessage = {
        id: response.id,
        type: 'agent',
        content: response.message || 'Operation completed',
        timestamp: response.timestamp,
        session_id: this.sessionId,
        response: response // This is the key fix!
      };

      this.addMessage(agentMessage);
    }
  }

  async sendApproval(approvalId: string, approved: boolean, feedback?: string): Promise<void> {
    try {
      const response = await this.http.post<AgentResponse>(`${this.apiUrl}/approval/${approvalId}`, {
        approval_id: approvalId,
        approved,
        feedback
      }).toPromise();

      if (response) {
        this.handleAgentResponse(response);
      }
    } catch (error) {
      this.handleError(error);
    }
  }

  async getMemoryStatus(): Promise<AgentResponse> {
    try {
      const response = await this.http.get<AgentResponse>(`${this.apiUrl}/memory/${this.sessionId}`).toPromise();
      return response!;
    } catch (error) {
      throw error;
    }
  }

  async clearMemory(): Promise<void> {
    try {
      await this.http.delete(`${this.apiUrl}/memory/${this.sessionId}`).toPromise();

      const clearMessage: ChatMessage = {
        id: this.generateId(),
        type: 'agent',
        content: 'Memory cleared successfully',
        timestamp: new Date().toISOString(),
        session_id: this.sessionId,
        response: null // Set to null for non-response messages
      };

      this.addMessage(clearMessage);
    } catch (error) {
      this.handleError(error);
    }
  }

  private addMessage(message: ChatMessage): void {
    this.messageHistory.push(message);
    this.messagesSubject.next([...this.messageHistory]);
  }

  private handleError(error: any): void {
    console.error('Chat service error:', error);

    const errorResponse: AgentResponse = {
      id: this.generateId(),
      success: false,
      type: 'error' as any,
      error: error.message || 'Something went wrong',
      session_id: this.sessionId,
      timestamp: new Date().toISOString()
    };

    const errorMessage: ChatMessage = {
      id: this.generateId(),
      type: 'agent',
      content: `Error: ${error.message || 'Something went wrong'}`,
      timestamp: new Date().toISOString(),
      session_id: this.sessionId,
      response: errorResponse // Store error response too
    };

    this.addMessage(errorMessage);
  }

  private generateId(): string {
    return 'msg-' + Math.random().toString(36).substr(2, 9);
  }

  clearChat(): void {
    this.messageHistory = [];
    this.messagesSubject.next([]);
  }

  createNewSession(): void {
    this.sessionId = this.generateSessionId();
    this.clearChat();
  }
}