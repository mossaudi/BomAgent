// src/app/components/table-renderer/table-renderer.component.ts
import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { UIRecommendation } from '../../models/chat.models';

@Component({
  selector: 'app-table-renderer',
  template: `
    <div class="table-container" *ngIf="tableData && tableData.length > 0">

      <!-- Table Controls -->
      <div class="table-controls">
        <div class="filters">
          <input
            type="text"
            class="filter-input"
            placeholder="Filter components..."
            [(ngModel)]="filterText"
            (input)="applyFilter()">
        </div>

        <div class="actions">
          <button
            *ngFor="let action of recommendations?.actions || []"
            class="btn btn-sm"
            (click)="onAction(action)">
            {{formatAction(action)}}
          </button>
        </div>
      </div>

      <!-- Table -->
      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th *ngFor="let col of displayColumns"
                  class="sortable"
                  (click)="sort(col)">
                {{formatColumnName(col)}}
                <span *ngIf="sortColumn === col" class="sort-indicator">
                  {{sortDirection === 'asc' ? '↑' : '↓'}}
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let row of filteredData; trackBy: trackByIndex">
              <td *ngFor="let col of displayColumns">
                <span [ngSwitch]="getColumnType(col, row[col])">

                  <!-- Confidence Score -->
                  <div *ngSwitchCase="'confidence'" class="confidence-bar">
                    <div class="confidence-fill"
                         [style.width.%]="(row[col] * 100)"
                         [ngClass]="getConfidenceClass(row[col])">
                    </div>
                    <span class="confidence-text">{{(row[col] * 100).toFixed(0)}}%</span>
                  </div>

                  <!-- URL -->
                  <a *ngSwitchCase="'url'" [href]="row[col]" target="_blank">
                    {{row[col] | slice:0:30}}{{row[col].length > 30 ? '...' : ''}}
                  </a>

                  <!-- Default -->
                  <span *ngSwitchDefault [title]="row[col]">
                    {{formatCellValue(row[col])}}
                  </span>

                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Table Summary -->
      <div class="table-summary">
        <span>Showing {{filteredData.length}} of {{tableData.length}} components</span>
        <span *ngIf="filterText">| Filtered by: "{{filterText}}"</span>
      </div>
    </div>
  `,
  styleUrls: ['./table-renderer.component.scss']
})
export class TableRendererComponent implements OnInit {
  @Input() data: any[] = [];
  @Input() recommendations: UIRecommendation | null = null;
  @Output() actionClicked = new EventEmitter<string>();

  tableData: any[] = [];
  filteredData: any[] = [];
  displayColumns: string[] = [];
  filterText = '';
  sortColumn = '';
  sortDirection: 'asc' | 'desc' = 'asc';

  ngOnInit(): void {
    this.initializeTable();
  }

  private initializeTable(): void {
    this.tableData = Array.isArray(this.data) ? this.data : [];
    this.filteredData = [...this.tableData];

    // Set display columns
    if (this.recommendations?.columns) {
      this.displayColumns = this.recommendations.columns;
    } else if (this.tableData.length > 0) {
      this.displayColumns = Object.keys(this.tableData[0]);
    }

    // Limit columns for better display
    if (this.displayColumns.length > 8) {
      this.displayColumns = this.displayColumns.slice(0, 8);
    }
  }

  applyFilter(): void {
    if (!this.filterText.trim()) {
      this.filteredData = [...this.tableData];
      return;
    }

    const filter = this.filterText.toLowerCase();
    this.filteredData = this.tableData.filter(row =>
      Object.values(row).some(value =>
        String(value).toLowerCase().includes(filter)
      )
    );
  }

  sort(column: string): void {
    if (this.sortColumn === column) {
      this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortColumn = column;
      this.sortDirection = 'asc';
    }

    this.filteredData.sort((a, b) => {
      const aVal = a[column];
      const bVal = b[column];

      let comparison = 0;
      if (aVal < bVal) comparison = -1;
      else if (aVal > bVal) comparison = 1;

      return this.sortDirection === 'asc' ? comparison : -comparison;
    });
  }

  onAction(action: string): void {
    this.actionClicked.emit(action);
  }

  formatAction(action: string): string {
    return action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  formatColumnName(col: string): string {
    return col.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  formatCellValue(value: any): string {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'string' && value.length > 50) {
      return value.substring(0, 47) + '...';
    }
    return String(value);
  }

  getColumnType(column: string, value: any): string {
    if (column.includes('confidence') || column.includes('rating')) return 'confidence';
    if (typeof value === 'string' && (value.startsWith('http') || value.includes('www'))) return 'url';
    return 'default';
  }

  getConfidenceClass(confidence: number): string {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
  }

  trackByIndex(index: number): number {
    return index;
  }
}