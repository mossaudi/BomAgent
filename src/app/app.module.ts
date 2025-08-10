// src/app/app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

// Components
import { AppComponent } from './app.component';
import { ChatInterfaceComponent } from './components/chat-interface/chat-interface.component';
import { ResponseRendererComponent } from './components/response-renderer/response-renderer.component';
import { TableRendererComponent } from './components/table-renderer/table-renderer.component';
import { TreeRendererComponent } from './components/tree-renderer/tree-renderer.component';
import { StatusRendererComponent } from './components/status-renderer/status-renderer.component';
import { JsonRendererComponent } from './components/json-renderer/json-renderer.component';
import { ApprovalPanelComponent } from './components/approval-panel/approval-panel.component';

// Services
import { ChatService } from './services/chat.service';

// Pipes
import { SlicePipe } from '@angular/common';

@NgModule({
  declarations: [
    AppComponent,
    ChatInterfaceComponent,
    ResponseRendererComponent,
    TableRendererComponent,
    TreeRendererComponent,
    StatusRendererComponent,
    JsonRendererComponent,
    ApprovalPanelComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    BrowserAnimationsModule
  ],
  providers: [
    ChatService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }