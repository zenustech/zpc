#pragma once
#include "zensim/ZpcImplPattern.hpp"
#include "zensim/ui/Widget.hpp"

namespace zs {

  using EventCategoryType = u32;
  enum event_e : EventCategoryType { event_unknown = 0, event_gui = 10 };
  struct ZsEvent {
    virtual ~ZsEvent() = default;
    virtual ZsEvent *clone() const { return nullptr; }
    virtual event_e getEventType() const { return event_unknown; }
  };
  /// @ref https://doc.qt.io/archives/qq/qq11-events.html
  using GuiEventType = u32;
  enum gui_event_e : GuiEventType {
    gui_event_unknown = 0,
    // mouse
    gui_event_mousePressed,
    gui_event_mouseReleased,
    gui_event_mouseDoubleClicked,
    gui_event_mouseScroll,
    gui_event_mouseMoved,
    // keyboard
    gui_event_keyPressed,
    gui_event_keyReleased,
    // user customization
    gui_event_user = 8192,
    gui_event_user_max = 32768,
  };
  struct GuiEvent : virtual ZsEvent {
    ~GuiEvent() override = default;
    GuiEvent *clone() const override { return nullptr; }
    virtual gui_event_e getType() const { return gui_event_unknown; }

    void accept() { _accepted = true; }
    void ignore() { _accepted = false; }
    bool isAccepted() const noexcept { return _accepted; }

  private:
    bool _accepted{false};
  };

  using GuiWidgetType = u32;
  enum gui_widget_e : GuiWidgetType {
    gui_widget_unknown = 0,
    gui_widget_item,
    gui_widget_item_image,
    gui_widget_item_button,
    gui_widget_item_checkbox,
    gui_widget_item_listbox,
    gui_widget_item_combo,
    gui_widget_item_text,
    gui_widget_item_slider,
    gui_widget_item_tree,
    gui_widget_item_table,
    gui_widget_item_tab,
    gui_widget_group,
    gui_widget_child_window,
    gui_widget_window,
    // user customization
    gui_window_user = 8192,
    gui_window_user_max = 32768,
  };
  struct WidgetConcept : virtual HierarchyConcept, virtual ObjectConcept {
    virtual ~WidgetConcept() = default;

    virtual bool onEvent(GuiEvent *e) { return true; }
    virtual bool eventFilter(ObjectConcept *obj, GuiEvent *e) { return false; }
    virtual void paint() = 0;
    virtual gui_widget_e getType() const { return gui_widget_unknown; }
  };
  struct EmptyWidget : WidgetConcept {
    ~EmptyWidget() override = default;
    void paint() override {}
  };
  struct WindowConcept : virtual WidgetConcept {
    virtual ~WindowConcept() = default;

    virtual gui_widget_e getType() const { return gui_widget_window; }
    virtual void placeAt(u32 layoutNodeId) {}
  };

}  // namespace zs