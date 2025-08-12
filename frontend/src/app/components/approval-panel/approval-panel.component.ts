// src/app/components/approval-panel/approval-panel.component.ts
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { FormControl } from '@angular/forms';
import { HumanApprovalRequest } from '../../models/chat.models';

@Component({
  selector: 'app-approval-panel',
  template: `
    <div class="approval-panel" *ngIf="approvalRequest">
      <div class="approval-header">
        <h4>🤖 Human Approval Required</h4>
        <span class="approval-step">Step: {{approvalRequest.step}}</span>
      </div>

      <div class="approval-content">
        <p class="approval-message">{{approvalRequest.message}}</p>

        <!-- Data Preview -->
        <div *ngIf="approvalRequest.data" class="data-preview">
          <h5>Preview:</h5>
          <app-json-renderer [data]="approvalRequest.data" [compact]="true"></app-json-renderer>
        </div>

        <!-- Feedback -->
        <div class="feedback-section">
          <label>Optional Feedback:</label>
          <textarea
            [formControl]="feedbackControl"
            placeholder="Add any comments or modifications..."
            rows="3">
          </textarea>
        </div>

        <!-- Actions -->
        <div class="approval-actions">
          <button
            class="btn btn-success"
            (click)="approve()">
            ✅ Approve
          </button>
          <button
            class="btn btn-danger"
            (click)="reject()">
            ❌ Reject
          </button>
          <button
            class="btn btn-secondary"
            (click)="modify()">
            ✏️ Modify
          </button>
        </div>

        <!-- Timeout Warning -->
        <div class="timeout-warning" *ngIf="timeRemaining > 0">
          <small>⏰ This approval expires in {{formatTime(timeRemaining)}}</small>
        </div>
      </div>
    </div>
  `,
  styleUrls: ['./approval-panel.component.scss']
})
export class ApprovalPanelComponent implements OnInit, OnDestroy {
  @Input() approvalRequest: HumanApprovalRequest | null = null;
  @Output() approved = new EventEmitter<HumanApprovalRequest>();
  @Output() rejected = new EventEmitter<HumanApprovalRequest>();
  @Output() modified = new EventEmitter<{ request: HumanApprovalRequest, feedback: string }>();

  feedbackControl = new FormControl('');
  timeRemaining = 0;
  private timer?: number;

  ngOnInit(): void {
    if (this.approvalRequest) {
      this.startTimer();
    }
  }

  ngOnDestroy(): void {
    if (this.timer) {
      clearInterval(this.timer);
    }
  }

  approve(): void {
    if (this.approvalRequest) {
      this.approved.emit(this.approvalRequest);
      this.stopTimer();
    }
  }

  reject(): void {
    if (this.approvalRequest) {
      this.rejected.emit(this.approvalRequest);
      this.stopTimer();
    }
  }

  modify(): void {
    if (this.approvalRequest) {
      this.modified.emit({
        request: this.approvalRequest,
        feedback: this.feedbackControl.value || ''
      });
      this.stopTimer();
    }
  }

  private startTimer(): void {
    if (!this.approvalRequest) return;

    const createdAt = new Date(this.approvalRequest.created_at);
    const expiresAt = new Date(createdAt.getTime() + this.approvalRequest.timeout_seconds * 1000);

    this.timer = window.setInterval(() => {
      const now = new Date();
      this.timeRemaining = Math.max(0, expiresAt.getTime() - now.getTime());

      if (this.timeRemaining <= 0) {
        this.stopTimer();
      }
    }, 1000);
  }

  private stopTimer(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }

  formatTime(milliseconds: number): string {
    const minutes = Math.floor(milliseconds / 60000);
    const seconds = Math.floor((milliseconds % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }
}