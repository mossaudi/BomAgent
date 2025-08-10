// src/app/components/json-renderer/json-renderer.component.ts
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-json-renderer',
  template: `
    <div class="json-container" [ngClass]="{'compact': compact}">
      <div class="json-header" *ngIf="!compact">
        <span>📄 Data</span>
        <button class="btn btn-xs" (click)="toggleExpanded()">
          {{expanded ? 'Collapse' : 'Expand'}}
        </button>
      </div>

      <pre class="json-content"
           [ngClass]="{'expanded': expanded || compact}"
           [innerHTML]="formattedJson">
      </pre>
    </div>
  `,
  styleUrls: ['./json-renderer.component.scss']
})
export class JsonRendererComponent {
  @Input() data: any = null;
  @Input() compact = false;

  expanded = false;
  formattedJson = '';

  ngOnChanges(): void {
    this.formatJson();
  }

  toggleExpanded(): void {
    this.expanded = !this.expanded;
  }

  private formatJson(): void {
    if (!this.data) {
      this.formattedJson = 'No data';
      return;
    }

    try {
      const jsonStr = JSON.stringify(this.data, null, 2);

      if (this.compact && jsonStr.length > 200) {
        this.formattedJson = this.syntaxHighlight(jsonStr.substring(0, 197) + '...');
      } else {
        this.formattedJson = this.syntaxHighlight(jsonStr);
      }
    } catch (error) {
      this.formattedJson = 'Invalid JSON data';
    }
  }

  private syntaxHighlight(json: string): string {
    return json
      .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        (match) => {
          let cls = 'json-number';
          if (/^"/.test(match)) {
            if (/:$/.test(match)) {
              cls = 'json-key';
            } else {
              cls = 'json-string';
            }
          } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
          } else if (/null/.test(match)) {
            cls = 'json-null';
          }
          return '<span class="' + cls + '">' + match + '</span>';
        });
  }
}