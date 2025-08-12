// src/app/app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { CommonModule } from '@angular/common';

// Components
import { AppComponent } from './app.component';
import { ChatInterfaceComponent } from './components/chat-interface/chat-interface.component';
import { ResponseRendererComponent } from './components/response-renderer/response-renderer.component';
import { TableRendererComponent } from './components/table-renderer/table-renderer.component';
import { TreeRendererComponent } from './components/tree-renderer/tree-renderer.component';
import { StatusRendererComponent } from './components/status-renderer/status-renderer.component';
import { JsonRendererComponent } from './components/json-renderer/json-renderer.component';
import { ApprovalPanelComponent } from './components/approval-panel/approval-panel.component';

// Services & Interceptors
import { ChatService } from './services/chat.service';
import { ErrorInterceptor } from './core/interceptors/error.interceptor';

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
    CommonModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    BrowserAnimationsModule
  ],
  providers: [
    ChatService,
    {
      provide: HTTP_INTERCEPTORS,
      useClass: ErrorInterceptor,
      multi: true
    }
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }