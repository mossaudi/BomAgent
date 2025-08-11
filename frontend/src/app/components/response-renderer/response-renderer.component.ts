// src/app/components/response-renderer/response-renderer.component.ts
import { Component, Input, Output, EventEmitter, OnChanges, SimpleChanges } from '@angular/core';
import { AgentResponse, ResponseType, UIRecommendation } from '../../models/chat.models';

@Component({
  selector: 'app-response-renderer',
  template: `
    <div class="response-container" *ngIf="response">

      <!-- Error Response -->
      <div *ngIf="response.response_type === 'error'" class="error-response">
        <div class="error-header">
          <i class="icon-error">❌</i>
          <span>Error</span>
        </div>
        <div class="error-content">{{response.error}}</div>
      </div>

      <!-- Success Responses -->
      <div *ngIf="response.success && response.response_type !== 'error'" class="success-response">

        <!-- Message -->
        <div *ngIf="response.message" class="response-message">
          {{response.message}}
        </div>

        <!-- Data Renderer -->
        <div class="data-container" [ngSwitch]="response.response_type">

          <!-- Table View -->
          <app-table-renderer
            *ngSwitchCase="'table'"
            [data]="response.data"
            [recommendations]="response.ui_recommendations"
            (actionClicked)="onActionClick($event)">
          </app-table-renderer>

          <!-- Tree View -->
          <app-tree-renderer
            *ngSwitchCase="'tree'"
            [data]="response.data"
            [recommendations]="response.ui_recommendations"
            (actionClicked)="onActionClick($event)">
          </app-tree-renderer>

          <!-- Status View -->
          <app-status-renderer
            *ngSwitchCase="'status'"
            [data]="response.data"
            [recommendations]="response.ui_recommendations"
            (actionClicked)="onActionClick($event)">
          </app-status-renderer>

          <!-- Generic/JSON View -->
          <app-json-renderer
            *ngSwitchDefault
            [data]="response.data">
          </app-json-renderer>

        </div>

        <!-- Actions -->
        <div *ngIf="response.next_actions && response.next_actions.length > 0" class="action-buttons">
          <button
            *ngFor="let action of response.next_actions"
            class="btn btn-sm btn-outline"
            (click)="onActionClick(action)">
            {{formatActionName(action)}}
          </button>
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
      console.log('Response received:', this.response);
    }
  }

  onActionClick(action: string): void {
    this.actionClicked.emit(action);
  }

  formatActionName(action: string): string {
    return action.replace(/_/g, ' ')
                 .replace(/\b\w/g, l => l.toUpperCase());
  }
}
