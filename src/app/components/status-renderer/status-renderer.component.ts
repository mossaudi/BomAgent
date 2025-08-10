// src/app/components/status-renderer/status-renderer.component.ts
import { Component, Input, Output, EventEmitter } from '@angular/core';
import { UIRecommendation } from '../../models/chat.models';

@Component({
  selector: 'app-status-renderer',
  template: `
    <div class="status-container" *ngIf="data">

      <div class="status-grid">
        <!-- Session Info -->
        <div class="status-card" *ngIf="data.session_id">
          <div class="card-header">
            <h4>📱 Session</h4>
          </div>
          <div class="card-content">
            <div class="status-item">
              <span class="label">ID:</span>
              <span class="value">{{data.session_id | slice:0:12}}...</span>
            </div>
            <div class="status-item" *ngIf="data.last_activity">
              <span class="label">Last Activity:</span>
              <span class="value">{{data.last_activity | date:'medium'}}</span>
            </div>
          </div>
        </div>

        <!-- Component Info -->
        <div class="status-card" *ngIf="data.stored_components !== undefined">
          <div class="card-header">
            <h4>🔧 Components</h4>
          </div>
          <div class="card-content">
            <div class="status-item">
              <span class="label">Stored:</span>
              <span class="value highlight">{{data.stored_components}}</span>
            </div>
            <div class="status-item" *ngIf="data.component_count !== undefined">
              <span class="label">Total:</span>
              <span class="value">{{data.component_count}}</span>
            </div>
          </div>
        </div>

        <!-- Memory Usage -->
        <div class="status-card" *ngIf="data.storage_usage">
          <div class="card-header">
            <h4>💾 Memory</h4>
          </div>
          <div class="card-content">
            <div class="status-item">
              <span class="label">Keys:</span>
              <span class="value">{{data.storage_usage.total_keys}}</span>
            </div>
            <div class="status-item">
              <span class="label">Size:</span>
              <span class="value">{{data.storage_usage.storage_size_kb}} KB</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Memory Keys -->
      <div class="memory-keys" *ngIf="data.memory_keys && data.memory_keys.length > 0">
        <h4>🗂️ Memory Contents</h4>
        <div class="key-list">
          <span *ngFor="let key of data.memory_keys" class="key-tag">
            {{formatKeyName(key)}}
          </span>
        </div>
      </div>

      <!-- Actions -->
      <div class="status-actions">
        <button
          *ngFor="let action of recommendations?.actions || ['refresh', 'clear']"
          class="btn btn-sm"
          (click)="onAction(action)">
          {{formatAction(action)}}
        </button>
      </div>
    </div>
  `,
  styleUrls: ['./status-renderer.component.scss']
})
export class StatusRendererComponent {
  @Input() data: any = null;
  @Input() recommendations: UIRecommendation | null = null;
  @Output() actionClicked = new EventEmitter<string>();

  onAction(action: string): void {
    this.actionClicked.emit(action);
  }

  formatAction(action: string): string {
    return action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  formatKeyName(key: string): string {
    // Remove session ID prefix and format
    const parts = key.split(':');
    const keyName = parts[parts.length - 1];
    return keyName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
}