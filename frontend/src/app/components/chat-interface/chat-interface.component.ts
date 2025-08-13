// src/app/components/chat-interface/chat-interface.component.ts - UPDATED
import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { FormControl, Validators } from '@angular/forms';
import { Subscription } from 'rxjs';
import { ChatService } from '../../services/chat.service';
import { ChatMessage, HumanApprovalRequest, AgentResponse } from '../../models/chat.models';

@Component({
  selector: 'app-chat-interface',
  template: `
    <div class="chat-container">
      <!-- Header -->
      <div class="chat-header">
        <h2>🤖 BOM Agent</h2>
        <div class="session-info">
          <span>Session: {{sessionId | slice:0:8}}...</span>
          <button class="btn btn-sm" (click)="showMemoryStatus()">Memory</button>
          <button class="btn btn-sm" (click)="createNewSession()">New Session</button>
        </div>
      </div>

      <!-- Messages -->
      <div class="messages-container" #messagesContainer>
        <div *ngFor="let message of messages"
             class="message"
             [ngClass]="'message-' + message.type">

          <div class="message-header">
            <span class="message-type">{{message.type | titlecase}}</span>
            <span class="message-time">{{message.timestamp | date:'short'}}</span>
          </div>

          <div class="message-content">
            <ng-container [ngSwitch]="message.type">

              <!-- User Message -->
              <div *ngSwitchCase="'user'" class="user-message">
                {{message.content}}
              </div>

              <!-- Agent Message -->
              <div *ngSwitchCase="'agent'" class="agent-message">
                <app-response-renderer
                  [response]="message.response"
                  (actionClicked)="onActionClicked($event)">
                </app-response-renderer>

                <!-- Fallback for messages without response data -->
                <div *ngIf="!message.response" class="simple-message">
                  {{message.content}}
                </div>
              </div>

              <!-- Approval Message -->
              <div *ngSwitchCase="'approval'" class="approval-message">
                <app-approval-panel
                  [approvalRequest]="pendingApproval"
                  (approved)="onApprovalResponse($event, true)"
                  (rejected)="onApprovalResponse($event, false)">
                </app-approval-panel>
              </div>

            </ng-container>
          </div>
        </div>

        <!-- Loading indicator -->
        <div *ngIf="isLoading" class="message message-agent">
          <div class="loading-indicator">
            <div class="spinner"></div>
            <span>Agent is thinking...</span>
          </div>
        </div>
      </div>

      <!-- Input -->
      <div class="chat-input">
        <div class="input-group">
          <input
            type="text"
            [formControl]="messageControl"
            placeholder="Ask me to analyze schematics, create BOMs, or manage components..."
            (keydown.enter)="sendMessage()"
            [disabled]="isLoading">

          <button
            class="btn btn-primary"
            (click)="sendMessage()"
            [disabled]="isLoading || messageControl.invalid">
            Send
          </button>
        </div>

        <!-- Quick Actions -->
        <div class="quick-actions">
          <button class="btn btn-sm btn-outline" (click)="quickAction('memory status')">
            📊 Memory Status
          </button>
          <button class="btn btn-sm btn-outline" (click)="quickAction('show BOMs')">
            📋 List BOMs
          </button>
          <button class="btn btn-sm btn-outline" (click)="quickAction('help')">
            ❓ Help
          </button>
          <!-- Add suggestions from last response -->
          <button *ngFor="let suggestion of getLastSuggestions()"
                  class="btn btn-sm btn-outline"
                  (click)="quickAction(suggestion)">
            ✨ {{suggestion}}
          </button>
        </div>
      </div>
    </div>
  `,
  styleUrls: ['./chat-interface.component.scss']
})
export class ChatInterfaceComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('messagesContainer') messagesContainer!: ElementRef;

  messages: ChatMessage[] = [];
  messageControl = new FormControl('', [Validators.required, Validators.minLength(1)]);
  isLoading = false;
  sessionId = '';
  pendingApproval: HumanApprovalRequest | null = null;

  private subscriptions = new Subscription();
  private shouldScrollToBottom = false;

  constructor(private chatService: ChatService) {}

  ngOnInit(): void {
    this.sessionId = this.chatService.getSessionId();

    // Subscribe to messages
    this.subscriptions.add(
      this.chatService.messages$.subscribe(messages => {
        this.messages = messages;
        this.shouldScrollToBottom = true;
        console.log('Messages updated:', messages); // Debug log
      })
    );

    // Subscribe to approval requests
    this.subscriptions.add(
      this.chatService.pendingApproval$.subscribe(approval => {
        this.pendingApproval = approval;
      })
    );
  }

  ngAfterViewChecked(): void {
    if (this.shouldScrollToBottom) {
      this.scrollToBottom();
      this.shouldScrollToBottom = false;
    }
  }

  ngOnDestroy(): void {
    this.subscriptions.unsubscribe();
  }

  async sendMessage(): Promise<void> {
    if (this.messageControl.valid && !this.isLoading) {
      const content = this.messageControl.value!.trim();
      if (content) {
        this.isLoading = true;
        this.messageControl.setValue('');

        try {
          await this.chatService.sendMessage(content);
        } finally {
          this.isLoading = false;
        }
      }
    }
  }

  async quickAction(action: string): Promise<void> {
    await this.chatService.sendMessage(action);
  }

  async onApprovalResponse(approval: HumanApprovalRequest, approved: boolean): Promise<void> {
    if (approval) {
      this.isLoading = true;
      try {
        await this.chatService.sendApproval(approval.id, approved);
        this.pendingApproval = null;
      } finally {
        this.isLoading = false;
      }
    }
  }

  onActionClicked(action: string): void {
    console.log('Action clicked:', action);

    switch (action) {
      case 'create_bom':
        this.quickAction('create BOM from components');
        break;
      case 'export':
        // Handle export
        break;
      case 'search_details':
        this.quickAction('search for more component details');
        break;
      default:
        this.quickAction(action);
    }
  }

  // REMOVED: No longer needed - response is stored in message.response
  // getResponseForMessage(message: ChatMessage): any {
  //   return message.response;
  // }

  // NEW: Get suggestions from the last agent response
  getLastSuggestions(): string[] {
    const lastAgentMessage = [...this.messages]
      .reverse()
      .find(msg => msg.type === 'agent' && msg.response);

    return lastAgentMessage?.response?.suggestions || [];
  }

  async showMemoryStatus(): Promise<void> {
    await this.quickAction('memory status');
  }

  createNewSession(): void {
    this.chatService.createNewSession();
    this.sessionId = this.chatService.getSessionId();
    this.pendingApproval = null;
  }

  private scrollToBottom(): void {
    try {
      const container = this.messagesContainer.nativeElement;
      container.scrollTop = container.scrollHeight;
    } catch (err) {
      console.error('Error scrolling to bottom:', err);
    }
  }
}