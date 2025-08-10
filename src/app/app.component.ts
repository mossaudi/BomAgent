// src/app/app.component.ts
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  template: `
    <div class="app-container">
      <app-chat-interface></app-chat-interface>
    </div>
  `,
  styles: [`
    .app-container {
      height: 100vh;
      width: 100vw;
    }
  `]
})
export class AppComponent {
  title = 'BOM Agent Chat';
}