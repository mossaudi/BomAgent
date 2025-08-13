// src/app/components/tree-renderer/tree-renderer.component.ts - UPDATED
import { Component, Input, Output, EventEmitter, OnChanges, SimpleChanges } from '@angular/core';
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
    <div class="tree-container" *ngIf="data">

      <!-- Tree Controls -->
      <div class="tree-controls">
        <div>
          <button class="btn btn-sm" (click)="expandAll()">Expand All</button>
          <button class="btn btn-sm" (click)="collapseAll()">Collapse All</button>
        </div>

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
            <span class="node-icon" *ngIf="hasChildren(node)">
              {{node.expanded ? '📂' : '📁'}}
            </span>
            <span class="node-icon" *ngIf="!hasChildren(node)">
              📄
            </span>

            <!-- Node Name -->
            <span class="node-name">{{node.name}}</span>

            <!-- Node Data Preview -->
            <span class="node-data" *ngIf="node.data && !hasChildren(node)">
              {{getDataPreview(node.data)}}
            </span>
            <span class="node-data" *ngIf="hasChildren(node)">
              ({{getDataSummary(node.data)}})
            </span>
          </div>

          <!-- Node Details -->
          <div class="node-details" *ngIf="node.expanded && node.data && !hasChildren(node)">
            <app-json-renderer [data]="node.data" [compact]="true"></app-json-renderer>
          </div>
        </div>
      </div>
    </div>
  `,
  styleUrls: ['./tree-renderer.component.scss']
})
export class TreeRendererComponent implements OnChanges {
  @Input() data: any = null;
  @Input() recommendations: UIRecommendation | null = null;
  @Output() actionClicked = new EventEmitter<string>();

  treeData: TreeNode[] = [];
  flattenedTree: TreeNode[] = [];

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data']) {
      console.log('Tree renderer data changed:', this.data);
      this.buildTree();
    }
  }

  // Helper method for template
  isObject(value: any): boolean {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
  }

  hasChildren(node: TreeNode): boolean {
    return !!(node.children && node.children.length > 0);
  }

  private buildTree(): void {
    if (!this.data) {
      this.treeData = [];
      this.flattenedTree = [];
      return;
    }

    console.log('Building tree from data:', this.data);
    this.treeData = this.convertToTree(this.data, 'Session Data');
    this.updateFlattenedTree();
  }

  private convertToTree(data: any, name = 'Root'): TreeNode[] {
    if (Array.isArray(data)) {
      return data.map((item, index) => ({
        name: `${name}[${index}]`,
        data: item,
        expanded: false,
        children: (this.isObject(item) || Array.isArray(item)) ?
          this.convertToTree(item, `${name}[${index}]`) : undefined
      }));
    } else if (this.isObject(data)) {
      return Object.keys(data).map(key => ({
        name: key,
        data: data[key],
        expanded: false,
        children: (this.isObject(data[key]) || Array.isArray(data[key])) ?
          this.convertToTree(data[key], key) : undefined
      }));
    } else {
      // Primitive value
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
    if (this.hasChildren(node)) {
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
      if (this.hasChildren(node)) {
        node.expanded = expanded;
        if (node.children) {
          this.setAllExpanded(node.children, expanded);
        }
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

  getDataPreview(data: any): string {
    if (data === null || data === undefined) return 'null';
    if (typeof data === 'string') {
      return data.length > 30 ? `"${data.substring(0, 27)}..."` : `"${data}"`;
    }
    if (typeof data === 'number' || typeof data === 'boolean') {
      return String(data);
    }
    return typeof data;
  }

  trackByNode(index: number, node: TreeNode): string {
    return `${node.name}_${node.level}_${index}`;
  }
}