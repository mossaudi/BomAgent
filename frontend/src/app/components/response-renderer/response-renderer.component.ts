// src/app/components/response-renderer/response-renderer.component.ts - WITH DEBUG
import { Component, Input, Output, EventEmitter, OnChanges, SimpleChanges } from '@angular/core';
import { AgentResponse, ResponseType, UIRecommendation } from '../../models/chat.models';

@Component({
  selector: 'app-response-renderer',
  template: `
    <div class="response-container" *ngIf="response">

      <!-- Debug Info (remove in production) -->
      <div style="background: #f0f0f0; padding: 0.5rem; margin: 0.5rem 0; font-size: 0.8em; border-radius: 4px;">
        <strong>Debug:</strong> Type: {{response.type}} | Success: {{response.success}} |
        Data: {{response.data ? 'Present' : 'None'}} | Message: {{response.message ? 'Present' : 'None'}}
      </div>

      <!-- Error Response -->
      <div *ngIf="!response.success || response.type === 'error'" class="error-response">
        <div class="error-header">
          <i class="icon-error">⚠️</i>
          <span>Error</span>
        </div>
        <div class="error-content">{{response.error || 'Unknown error occurred'}}</div>
      </div>

      <!-- Success Responses -->
      <div *ngIf="response.success && response.type !== 'error'" class="success-response">

        <!-- Message -->
        <div *ngIf="response.message" class="response-message">
          {{response.message}}
        </div>

        <!-- Data Renderer -->
        <div class="data-container" [ngSwitch]="response.type">

          <!-- Table View -->
          <div *ngSwitchCase="'table'">
            <h6>📊 Table View</h6>
            <app-table-renderer
              [data]="response.data"
              [recommendations]="response.ui_recommendations || null"
              (actionClicked)="onActionClick($event)">
            </app-table-renderer>
          </div>

          <!-- Tree View -->
          <div *ngSwitchCase="'tree'">
            <h6>🌳 Tree View</h6>
            <app-tree-renderer
              [data]="response.data"
              [recommendations]="response.ui_recommendations || null"
              (actionClicked)="onActionClick($event)">
            </app-tree-renderer>
          </div>

          <!-- Status View -->
          <div *ngSwitchCase="'status'">
            <h6>📊 Status View</h6>
            <app-status-renderer
              [data]="response.data"
              [recommendations]="response.ui_recommendations || null"
              (actionClicked)="onActionClick($event)">
            </app-status-renderer>
          </div>

          <!-- Generic/JSON View -->
          <div *ngSwitchDefault>
            <h6>📄 Data View ({{response.type}})</h6>
            <app-json-renderer [data]="response.data"></app-json-renderer>
          </div>

        </div>

        <!-- Suggestions -->
        <div *ngIf="response.suggestions && response.suggestions.length > 0" class="action-buttons">
          <h5>💡 Suggestions:</h5>
          <div class="buttons-grid">
            <button
              *ngFor="let suggestion of response.suggestions"
              class="btn btn-sm suggestion"
              (click)="onActionClick(suggestion)">
              ✨ {{suggestion}}
            </button>
          </div>
        </div>

        <!-- Next Actions -->
        <div *ngIf="response.next_actions && response.next_actions.length > 0" class="action-buttons">
          <h5>🎯 Actions:</h5>
          <div class="buttons-grid">
            <button
              *ngFor="let action of response.next_actions"
              class="btn btn-sm"
              (click)="onActionClick(action)">
              🔧 {{formatActionName(action)}}
            </button>
          </div>
        </div>

        <!-- Fallback if no data to display -->
        <div *ngIf="!response.data && !response.suggestions && !response.next_actions"
             style="padding: 1rem; background: #f8f9fa; border-radius: 6px; text-align: center; color: #6c757d;">
          <em>Response received but no data to display</em>
        </div>
      </div>
    </div>
  `,
  styleUrls: ['./response-renderer.component.scss']
})
export class ResponseRendererComponent implements OnChanges {
  @Input() response: AgentResponse | null = null;
  @Output() actionClicked = new EventEmitter<string>();

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['response'] && this.response) {
      console.log('🔍 Response renderer received:', this.response);
      console.log('🔍 Response type:', this.response.type);
      console.log('🔍 Response data:', this.response.data);
      console.log('🔍 Response success:', this.response.success);
      console.log('🔍 Response suggestions:', this.response.suggestions);
    }
  }

  onActionClick(action: string): void {
    console.log('🎯 Action clicked:', action);
    this.actionClicked.emit(action);
  }

  formatActionName(action: string): string {
    return action.replace(/_/g, ' ')
                 .replace(/\b\w/g, l => l.toUpperCase());
  }
}