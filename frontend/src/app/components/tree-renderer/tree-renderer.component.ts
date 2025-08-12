// src/app/components/tree-renderer/tree-renderer.component.ts
import { Component, Input, Output, EventEmitter } from '@angular/core';
import { UIRecommendation } from '../../models/chat.models';

interface TreeNode {
  name: string;
  children?: TreeNode[];
  data?: any;
  expanded?: boolean;
  level?: number;
}

@Component({
  selector: 'app-tree-renderer',
  template: `
    <div class="tree-container" *ngIf="treeData">

      <!-- Tree Controls -->
      <div class="tree-controls">
        <button class="btn btn-sm" (click)="expandAll()">Expand All</button>
        <button class="btn btn-sm" (click)="collapseAll()">Collapse All</button>

        <div class="tree-actions">
          <button
            *ngFor="let action of recommendations?.actions || []"
            class="btn btn-sm"
            (click)="onAction(action)">
            {{formatAction(action)}}
          </button>
        </div>
      </div>

      <!-- Tree Structure -->
      <div class="tree-structure">
        <div *ngFor="let node of flattenedTree; trackBy: trackByNode"
             class="tree-node"
             [style.padding-left.px]="(node.level || 0) * 20">

          <div class="node-content" (click)="toggleNode(node)">

            <!-- Expand/Collapse Icon -->
            <span class="node-icon" *ngIf="node.children && node.children.length > 0">
              {{node.expanded ? '📂' : '📁'}}
            </span>
            <span class="node-icon" *ngIf="!node.children || node.children.length === 0">
              📄
            </span>

            <!-- Node Name -->
            <span class="node-name">{{node.name}}</span>

            <!-- Node Data -->
            <span class="node-data" *ngIf="node.data && isObject(node.data)">
              ({{getDataSummary(node.data)}})
            </span>
          </div>

          <!-- Node Details -->
          <div class="node-details" *ngIf="node.expanded && node.data">
            <app-json-renderer [data]="node.data" [compact]="true"></app-json-renderer>
          </div>
        </div>
      </div>
    </div>
  `,
  styleUrls: ['./tree-renderer.component.scss']
})
export class TreeRendererComponent {
  @Input() data: any = null;
  @Input() recommendations: UIRecommendation | null = null;
  @Output() actionClicked = new EventEmitter<string>();

  treeData: TreeNode[] = [];
  flattenedTree: TreeNode[] = [];

  ngOnChanges(): void {
    this.buildTree();
  }

  // Helper method for template
  isObject(value: any): boolean {
    return typeof value === 'object' && value !== null;
  }

  private buildTree(): void {
    if (!this.data) return;

    this.treeData = this.convertToTree(this.data);
    this.updateFlattenedTree();
  }

  private convertToTree(data: any, name = 'Root'): TreeNode[] {
    if (Array.isArray(data)) {
      return data.map((item, index) => ({
        name: `Item ${index + 1}`,
        data: item,
        expanded: false,
        children: this.isObject(item) ? this.convertToTree(item, `Item ${index + 1}`) : undefined
      }));
    } else if (this.isObject(data)) {
      return Object.keys(data).map(key => ({
        name: key,
        data: data[key],
        expanded: false,
        children: this.isObject(data[key]) ? this.convertToTree(data[key], key) : undefined
      }));
    } else {
      return [{
        name: name,
        data: data,
        expanded: false
      }];
    }
  }

  private updateFlattenedTree(): void {
    this.flattenedTree = [];
    this.flattenTree(this.treeData, 0);
  }

  private flattenTree(nodes: TreeNode[], level: number): void {
    for (const node of nodes) {
      node.level = level;
      this.flattenedTree.push(node);

      if (node.expanded && node.children) {
        this.flattenTree(node.children, level + 1);
      }
    }
  }

  toggleNode(node: TreeNode): void {
    if (node.children && node.children.length > 0) {
      node.expanded = !node.expanded;
      this.updateFlattenedTree();
    }
  }

  expandAll(): void {
    this.setAllExpanded(this.treeData, true);
    this.updateFlattenedTree();
  }

  collapseAll(): void {
    this.setAllExpanded(this.treeData, false);
    this.updateFlattenedTree();
  }

  private setAllExpanded(nodes: TreeNode[], expanded: boolean): void {
    for (const node of nodes) {
      node.expanded = expanded;
      if (node.children) {
        this.setAllExpanded(node.children, expanded);
      }
    }
  }

  onAction(action: string): void {
    this.actionClicked.emit(action);
  }

  formatAction(action: string): string {
    return action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  getDataSummary(data: any): string {
    if (Array.isArray(data)) {
      return `${data.length} items`;
    } else if (this.isObject(data)) {
      return `${Object.keys(data).length} properties`;
    } else {
      return typeof data;
    }
  }

  trackByNode(index: number, node: TreeNode): string {
    return `${node.name}_${node.level}_${index}`;
  }
}