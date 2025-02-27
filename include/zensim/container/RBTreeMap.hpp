#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stack>
#include <utility>
#include <vector>

#include "zensim/types/Iterator.h"

namespace zs {
  /**
   * A red–black tree (RBTree) is a kind of self-balancing binary search tree.
   * Each node stores an extra field representing "color" (RED or BLACK), used
   * to ensure that the tree remains balanced during insertions and deletions.
   *
   * @tparam Key the type of keys maintained by this map
   * @tparam Value the type of mapped values
   * @tparam Compare the compare function
   */
  template <typename Key, typename Value, typename Compare = std::less<Key> > class RBTreeMap {
  private:
    Compare compare = Compare();

    struct Node {
      enum { RED, BLACK } color = RED;
      Node* parent = nullptr;
      Node* left = nullptr;
      Node* right = nullptr;
      Key first{};
      Value second{};

      Node() = default;
      explicit Node(Key k) : first(std::move(k)) {}
      explicit Node(Key k, Value v) : first(std::move(k)), second(std::move(v)) {}
      ~Node() = default;

      inline bool isRed() const noexcept { return this->color == RED; }
      inline bool isBlack() const noexcept { return this->color == BLACK; }

      inline void release() noexcept {
        this->parent = nullptr;
        if (this->left != nullptr) {
          if (this->left != this) {
            this->left->release();
            delete this->left;
          }
          this->left = nullptr;
        }
        if (this->right != nullptr) {
          if (this->right != this) {
            this->right->release();
            delete this->right;
          }
        }
      }
    };

    enum Direction { LEFT = -1, ROOT = 0, RIGHT = 1 };
    // using NodeProvider = const std::function<Node*(void)> &;
    // using NodeConsumer = const std::function<void(const Node*)> &;

    Node* header = nullptr;
    Node* root = nullptr;
    size_t cnt = 0;

    inline bool isRoot(const Node* node) const noexcept { return this->root == node; }

    inline Direction getDirection(const Node* node) const noexcept {
      if (!isRoot(node)) {
        if (node == node->parent->left) {
          return Direction::LEFT;
        } else {
          return Direction::RIGHT;
        }
      } else {
        return Direction::ROOT;
      }
    }

    inline Node* getSibling(const Node* node) const noexcept {
      if (getDirection(node) == LEFT) {
        return node->parent->right;
      } else {
        return node->parent->left;
      }
    }

    inline bool hasSibling(const Node* node) const noexcept {
      return !isRoot(node) && getSibling(node) != nullptr;
    }

    inline Node* getUncle(const Node* node) const noexcept { return getSibling(node->parent); }

    inline bool hasUncle(const Node* node) const noexcept {
      return !isRoot(node) && hasSibling(node->parent);
    }

    inline Node* getGrandParent(const Node* node) const noexcept { return node->parent->parent; }

    inline bool hasGrandParent(const Node* node) const noexcept {
      return !isRoot(node) && !isRoot(node->parent);
    }

    using K = const Key&;
    using V = const Value&;
    using reference = Node&;
    using const_reference = const Node&;

    struct iterator_impl : IteratorInterface<iterator_impl> {
      constexpr iterator_impl() = default;
      constexpr iterator_impl(Node* base) : _node{base} {}

      constexpr reference dereference() { return *_node; }
      constexpr bool equal_to(iterator_impl it) const noexcept { return it._node == _node; }
      constexpr void increment() noexcept {
        if (_node->right != nullptr) {
          _node = _node->right;
          while (_node->left != nullptr) _node = _node->left;
        } else {
          Node* _y = _node->parent;
          while (_node == _y->right) {
            _node = _y;
            _y = _y->parent;
          }
          if (_node->right != _y) _node = _y;
        }
      }
      constexpr void decrement() noexcept {
        if (_node->isRed() && _node->parent->parent == _node) {
          _node = _node->right;
        } else if (_node->left != nullptr) {
          _node = _node->left;
          while (_node->right != nullptr) _node = _node->right;
        } else {
          Node* _y = _node->parent;
          while (_node == _y->left) {
            _node = _y;
            _y = _y->parent;
          }
          if (_node->left != _y) _node = _y;
        }
      }

    protected:
      Node* _node{nullptr};
    };
    struct const_iterator_impl : IteratorInterface<const_iterator_impl> {
      constexpr const_iterator_impl() = default;
      constexpr const_iterator_impl(const Node* base) : _node{base} {}

      constexpr const_reference dereference() { return *_node; }
      constexpr bool equal_to(const_iterator_impl it) const noexcept { return it._node == _node; }
      constexpr void increment() noexcept {
        if (_node->right != nullptr) {
          _node = _node->right;
          while (_node->left != nullptr) _node = _node->left;
        } else {
          const Node* _y = _node->parent;
          while (_node == _y->right) {
            _node = _y;
            _y = _y->parent;
          }
          if (_node->right != _y) _node = _y;
        }
      }
      constexpr void decrement() noexcept {
        if (_node->isRed() && _node->parent->parent == _node) {
          _node = _node->right;
        } else if (_node->left != nullptr) {
          _node = _node->left;
          while (_node->right != nullptr) _node = _node->right;
        } else {
          const Node* _y = _node->parent;
          while (_node == _y->left) {
            _node = _y;
            _y = _y->parent;
          }
          if (_node->left != _y) _node = _y;
        }
      }

    protected:
      const Node* _node{nullptr};
    };
    struct reverse_iterator_impl : IteratorInterface<reverse_iterator_impl> {
      constexpr reverse_iterator_impl() = default;
      constexpr reverse_iterator_impl(Node* base) : _node{base} {}

      constexpr reference dereference() { return *_node; }
      constexpr bool equal_to(reverse_iterator_impl it) const noexcept { return it._node == _node; }
      constexpr void increment() noexcept {
        if (_node->left != nullptr) {
          _node = _node->left;
          while (_node->right != nullptr) _node = _node->right;
        } else {
          Node* _y = _node->parent;
          while (_node == _y->left) {
            _node = _y;
            _y = _y->parent;
          }
          if (_node->left != _y) _node = _y;
        }
      }
      constexpr void decrement() noexcept {
        if (_node->isRed() && _node->parent->parent == _node) {
          _node = _node->left;
        } else if (_node->right != nullptr) {
          _node = _node->right;
          while (_node->left != nullptr) _node = _node->left;
        } else {
          Node* _y = _node->parent;
          while (_node == _y->right) {
            _node = _y;
            _y = _y->parent;
          }
          if (_node->right != _y) _node = _y;
        }
      }

    protected:
      Node* _node{nullptr};
    };
    struct const_reverse_iterator_impl : IteratorInterface<const_reverse_iterator_impl> {
      constexpr const_reverse_iterator_impl() = default;
      constexpr const_reverse_iterator_impl(const Node* base) : _node{base} {}

      constexpr const_reference dereference() { return *_node; }
      constexpr bool equal_to(const_reverse_iterator_impl it) const noexcept {
        return it._node == _node;
      }
      constexpr void increment() noexcept {
        if (_node->left != nullptr) {
          _node = _node->left;
          while (_node->right != nullptr) _node = _node->right;
        } else {
          const Node* _y = _node->parent;
          while (_node == _y->left) {
            _node = _y;
            _y = _y->parent;
          }
          if (_node->left != _y) _node = _y;
        }
      }
      constexpr void decrement() noexcept {
        if (_node->isRed() && _node->parent->parent == _node) {
          _node = _node->right;
        } else if (_node->right != nullptr) {
          _node = _node->right;
          while (_node->left != nullptr) _node = _node->left;
        } else {
          const Node* _y = _node->parent;
          while (_node == _y->right) {
            _node = _y;
            _y = _y->parent;
          }
          if (_node->right != _y) _node = _y;
        }
      }

    protected:
      const Node* _node{nullptr};
    };

  public:
    // using KeyValueConsumer = const std::function<void(K, V)> &;
    // using MutKeyValueConsumer = const std::function<void(K, Value &)> &;
    // using KeyValueFilter = const std::function<bool(K, V)> &;
    using iterator = LegacyIterator<iterator_impl>;
    using const_iterator = LegacyIterator<const_iterator_impl>;
    using reverse_iterator = LegacyIterator<reverse_iterator_impl>;
    using const_reverse_iterator = LegacyIterator<const_reverse_iterator_impl>;

    class NoSuchMappingException : protected std::exception {
    private:
      const char* message;

    public:
      explicit NoSuchMappingException(const char* msg) : message(msg) {}

      const char* what() const noexcept override { return message; }
    };

    RBTreeMap() noexcept {
      this->header = new Node();
      this->header->parent = nullptr;
      this->header->left = this->header->right = this->header;
    }

    ~RBTreeMap() noexcept {
      this->clear();
      this->header->left = this->header->right = nullptr;
      delete this->header;
      this->header = nullptr;
    }

    constexpr auto begin() noexcept { return make_iterator<iterator_impl>(this->header->left); }
    constexpr auto end() noexcept { return make_iterator<iterator_impl>(this->header); }
    constexpr auto cbegin() const noexcept {
      return make_iterator<const_iterator_impl>(this->header->left);
    }
    constexpr auto cend() const noexcept {
      return make_iterator<const_iterator_impl>(this->header);
    }
    constexpr auto rbegin() noexcept {
      return make_iterator<reverse_iterator_impl>(this->header->right);
    }
    constexpr auto rend() noexcept { return make_iterator<reverse_iterator_impl>(this->header); }
    constexpr auto crbegin() const noexcept {
      return make_iterator<const_reverse_iterator_impl>(this->header->right);
    }
    constexpr auto crend() const noexcept {
      return make_iterator<const_reverse_iterator_impl>(this->header);
    }

    /**
     * Returns the number of entries in this map.
     * @return size_t
     */
    inline size_t size() const noexcept { return this->cnt; }

    /**
     * Returns true if this collection contains no elements.
     * @return bool
     */
    inline bool empty() const noexcept { return this->cnt == 0; }

    /**
     * Removes all of the elements from this map.
     */
    void clear() noexcept {
      this->header->left = this->header->right = this->header;
      if (this->root != nullptr) {
        this->root->release();
        this->header->parent = nullptr;
        delete this->root;
        this->root = nullptr;
      }
      this->cnt = 0;
    }

    /**
     * Returns the value to which the specified key is mapped; If this map
     * contains no mapping for the key, a {@code NoSuchMappingException} will
     * be thrown.
     * @param key
     * @return RBTreeMap<Key, Value>::Value
     * @throws NoSuchMappingException
     */
    Value get(K key) const {
      if (this->root == nullptr) {
        throw NoSuchMappingException("Invalid key");
      } else {
        const Node* node = this->getNode(this->root, key);
        if (node != nullptr) {
          return node->second;
        } else {
          throw NoSuchMappingException("Invalid key");
        }
      }
    }

    /**
     * Returns the value to which the specified key is mapped; If this map
     * contains no mapping for the key, a new mapping with a default value
     * will be inserted.
     * @param key
     * @return RBTreeMap<Key, Value>::Value &
     */
    Value& getOrDefault(K key) {
      if (this->root == nullptr) {
        this->root = new Node(key);
        this->root->parent = this->header;
        this->root->color = Node::BLACK;
        this->header->parent = this->root;
        this->header->left = this->header->right = this->root;
        this->cnt += 1;
        return this->root->second;
      } else {
        return this->getNodeOrProvide(this->root, key, [&key]() { return new Node(key); })->second;
      }
    }

    /**
     * Returns true if this map contains a mapping for the specified key.
     * @param key
     * @return size_t
     */
    size_t count(K key) const { return this->getNode(this->root, key) ? 1 : 0; }

    /**
     * Associates the specified value with the specified key in this map.
     * @param key
     * @param value
     */
    void insert(K key, V value) {
      if (this->root == nullptr) {
        this->root = new Node(key, value);
        this->root->parent = this->header;
        this->root->color = Node::BLACK;
        this->header->parent = this->root;
        this->header->left = this->header->right = this->root;
        this->cnt += 1;
      } else {
        this->insert(this->root, key, value);
      }
    }

    Value at(K key) const { return this->get(key); }
    Value& operator[](K key) { return this->getOrDefault(key); }

    /**
     * Removes the element at position
     * @param position iterator to the element to remove
     * @return RBTreeMap<Key, Value>::iterator
     */
    iterator erase(iterator position) {
      if (position == nullptr || position == this->end()) {
        return position;
      } else {
        Node* node = &*position;
        iterator next = make_iterator<iterator_impl>(node);
        ++next;
        this->remove(this->root, node->first, [](const Node*) {});
        return next;
      }
    }

    /**
     * Removes the element at position
     * @param position iterator to the element to remove
     * @return RBTreeMap<Key, Value>::iterator
     */
    iterator erase(const_iterator position) {
      if (position == nullptr || position == this->end()) {
        return position;
      } else {
        Node* node = &*position;
        iterator next = make_iterator<iterator_impl>(node);
        ++next;
        this->remove(this->root, node->first, [](const Node*) {});
        return next;
      }
    }

    /**
     * Removes the elements in the range [first; last)
     * @param first range of elements to remove
     * @param last range of elements to remove
     * @return RBTreeMap<Key, Value>::iterator
     */
    iterator erase(iterator first, iterator last) {
      if (first == this->begin() && last == this->end()) {
        this->clear();
        return this->end();
      } else {
        while (first != last) {
          iterator next = make_iterator<iterator_impl>(&*first);
          ++next;
          this->remove(this->root, first->first, [](const Node*) {});
          first = next;
        }
        return last;
      }
    }

    /**
     * Removes the elements in the range [first; last)
     * @param first range of elements to remove
     * @param last range of elements to remove
     * @return RBTreeMap<Key, Value>::iterator
     */
    iterator erase(const_iterator first, const_iterator last) {
      if (first == this->begin() && last == this->end()) {
        this->clear();
        return this->end();
      } else {
        iterator position = make_iterator<iterator_impl>(&*first);
        while (position != last) {
          iterator next = make_iterator<iterator_impl>(&*first);
          ++next;
          this->remove(this->root, position->first, [](const Node*) {});
          position = next;
        }
        return position;
      }
    }

    /**
     * Removes the elements with the key value key
     * @param key key value of the elements to remove
     * @return size_t
     */
    size_t erase(K key) {
      if (this->root == nullptr) {
        return 0;
      } else {
        return this->remove(this->root, key, [](const Node*) {}) ? 1 : 0;
      }
    }

    /**
     * Returns an iterator pointing to the first element that is not less than key.
     * @param key
     * @return RBTreeMap<Key, Value>::iterator
     */
    iterator lower_bound(K key) const {
      if (this->root == nullptr) {
        return make_iterator<iterator_impl>(this->header);
      }

      Node* node = this->root;

      while (node != nullptr) {
        if (key == node->first) {
          return make_iterator<iterator_impl>(node);
        }

        if (compare(key, node->first)) {
          /* key < node->first */
          if (node->left != nullptr) {
            node = node->left;
          } else {
            return make_iterator<iterator_impl>(node);
          }
        } else {
          /* key > node->first */
          if (node->right != nullptr) {
            node = node->right;
          } else {
            while (getDirection(node) == Direction::RIGHT) {
              if (node != nullptr) {
                node = node->parent;
              } else {
                make_iterator<iterator_impl>(this->header);
              }
            }
            if (node->parent == nullptr) {
              make_iterator<iterator_impl>(this->header);
            }
            return make_iterator<iterator_impl>(node->parent);
          }
        }
      }
      return make_iterator<iterator_impl>(this->header);
    }

    /**
     * Returns an iterator pointing to the first element that is greater than key.
     * @param key
     * @return RBTreeMap<Key, Value>::iterator
     */
    iterator upper_bound(K key) {
      if (this->root == nullptr) {
        return make_iterator<iterator_impl>(this->header);
      }

      Node* node = this->root;

      while (node != nullptr) {
        if (compare(key, node->first)) {
          /* key < node->first */
          if (node->left != nullptr) {
            node = node->left;
          } else {
            return make_iterator<iterator_impl>(node);
          }
        } else {
          /* key >= node->first */
          if (node->right != nullptr) {
            node = node->right;
          } else {
            while (getDirection(node) == Direction::RIGHT) {
              if (node != nullptr) {
                node = node->parent;
              } else {
                return make_iterator<iterator_impl>(this->header);
              }
            }
            if (node->parent == nullptr) {
              return make_iterator<iterator_impl>(this->header);
            }
            return make_iterator<iterator_impl>(node->parent);
          }
        }
      }
      return make_iterator<iterator_impl>(this->header);
    }

    /**
     * Remove all entries that satisfy the filter condition.
     * @param filter
     */
    template <typename KeyValueFilterF> void removeAll(KeyValueFilterF&& filter) {
      std::vector<Key> keys;
      this->inorderTraversal([&](const Node* node) {
        if (filter(node->first, node->second)) {
          keys.push_back(node->first);
        }
      });
      for (const Key& key : keys) {
        this->remove(key);
      }
    }

    /**
     * Performs the given action for each key and value entry in this map.
     * The value is immutable for the action.
     * @param action
     */
    template <typename KeyValueConsumerF> void forEach(KeyValueConsumerF&& action) const {
      this->inorderTraversal([&](const Node* node) { action(node->first, node->second); });
    }

    /**
     * Performs the given action for each key and value entry in this map.
     * The value is mutable for the action.
     * @param action
     */
    template <typename MutKeyValueConsumerF> void forEachMut(MutKeyValueConsumerF&& action) {
      this->inorderTraversal([&](const Node* node) { action(node->first, node->second); });
    }

  private:
    static void maintainRelationship(Node* node) {
      if (node->left != nullptr) {
        node->left->parent = node;
      }
      if (node->right != nullptr) {
        node->right->parent = node;
      }
    }

    void rotateLeft(Node* node) {
      //     |                       |
      //     N                       S
      //    / \     l-rotate(N)     / \
      //   L   S    ==========>    N   R
      //      / \                 / \
      //     M   R               L   M
      assert(node != nullptr && node->right != nullptr);
      Node* parent = node->parent;
      Direction direction = getDirection(node);

      Node* successor = node->right;
      node->right = successor->left;
      successor->left = node;

      maintainRelationship(node);
      maintainRelationship(successor);

      switch (direction) {
        case Direction::ROOT:
          this->root = this->header->parent = successor;
          break;
        case Direction::LEFT:
          parent->left = successor;
          break;
        case Direction::RIGHT:
          parent->right = successor;
          break;
      }

      successor->parent = parent;
    }

    void rotateRight(Node* node) {
      //       |                   |
      //       N                   S
      //      / \   r-rotate(N)   / \
      //     S   R  ==========>  L   N
      //    / \                     / \
      //   L   M                   M   R
      assert(node != nullptr && node->left != nullptr);
      Node* parent = node->parent;
      Direction direction = getDirection(node);

      Node* successor = node->left;
      node->left = successor->right;
      successor->right = node;

      maintainRelationship(node);
      maintainRelationship(successor);

      switch (direction) {
        case Direction::ROOT:
          this->root = this->header->parent = successor;
          break;
        case Direction::LEFT:
          parent->left = successor;
          break;
        case Direction::RIGHT:
          parent->right = successor;
          break;
      }

      successor->parent = parent;
    }

    inline void rotateSameDirection(Node* node, Direction direction) {
      assert(direction != Direction::ROOT);
      if (direction == Direction::LEFT) {
        rotateLeft(node);
      } else {
        rotateRight(node);
      }
    }

    inline void rotateOppositeDirection(Node* node, Direction direction) {
      assert(direction != Direction::ROOT);
      if (direction == Direction::LEFT) {
        rotateRight(node);
      } else {
        rotateLeft(node);
      }
    }

    void maintainAfterInsert(Node* node) {
      assert(node != nullptr);

      if (isRoot(node)) {
        // Case 1: Current node is root
        //  No need to fix.
        assert(node->isRed());
        node->color = Node::BLACK;
        return;
      }

      if (node->parent->isBlack()) {
        // Case 2: Parent is BLACK
        //  No need to fix.
        return;
      }

      if (isRoot(node->parent)) {
        // Case 3: Parent is root and is RED
        //   Paint parent to BLACK.
        //    <P>         [P]
        //     |   ====>   |
        //    <N>         <N>
        assert(node->parent->isRed());
        node->parent->color = Node::BLACK;
        return;
      }

      if (hasUncle(node) && getUncle(node)->isRed()) {
        // Case 4: Both parent and uncle are RED
        //   Paint parent and uncle to BLACK;
        //   Paint grandparent to RED.
        //        [G]             <G>
        //        / \             / \
        //      <P> <U>  ====>  [P] [U]
        //      /               /
        //    <N>             <N>
        assert(node->parent->isRed());
        node->parent->color = Node::BLACK;
        getUncle(node)->color = Node::BLACK;
        getGrandParent(node)->color = Node::RED;
        maintainAfterInsert(getGrandParent(node));
        return;
      }

      if (!hasUncle(node) || getUncle(node)->isBlack()) {
        // Case 5 & 6: Parent is RED and Uncle is BLACK
        //   p.s. NIL nodes are also considered BLACK

        if (getDirection(node) != getDirection(node->parent)) {
          // Case 5: Current node is the opposite direction as parent
          //   Step 1. If node is a LEFT child, perform l-rotate to parent;
          //           If node is a RIGHT child, perform r-rotate to parent.
          //   Step 2. Goto Case 6.
          //      [G]                 [G]
          //      / \    rotate(P)    / \
          //    <P> [U]  ========>  <N> [U]
          //      \                 /
          //      <N>             <P>
          Node* parent = node->parent;
          if (getDirection(node) == Direction::LEFT) {
            rotateRight(node->parent);
          } else {
            rotateLeft(node->parent);
          }
          node = parent;
        }

        // Case 6: Current node is the same direction as parent
        //   Step 1. If node is a LEFT child, perform r-rotate to grandparent;
        //           If node is a RIGHT child, perform l-rotate to grandparent.
        //   Step 2. Paint parent (before rotate) to BLACK;
        //           Paint grandparent (before rotate) to RED.
        //        [G]                 <P>               [P]
        //        / \    rotate(G)    / \    repaint    / \
        //      <P> [U]  ========>  <N> [G]  ======>  <N> <G>
        //      /                         \                 \
        //    <N>                         [U]               [U]
        if (getDirection(node->parent) == Direction::LEFT) {
          rotateRight(getGrandParent(node));
        } else {
          rotateLeft(getGrandParent(node));
        }
        node->parent->color = Node::BLACK;
        getSibling(node)->color = Node::RED;

        return;
      }
    }

    template <typename NodeProviderF>
    Node* getNodeOrProvide(Node*& node, K key, NodeProviderF&& provide) {
      assert(node != nullptr);

      if (key == node->first) {
        return node;
      }

      Node* result;

      if (compare(key, node->first)) {
        /* key < node->first */
        if (node->left == nullptr) {
          result = provide();
          node->left = result;
          node->left->parent = node;
          if (this->header->left == node) this->header->left = node->left;
          maintainAfterInsert(result);
          this->cnt += 1;
        } else {
          result = getNodeOrProvide(node->left, key, provide);
        }
      } else {
        /* key > node->first */
        if (node->right == nullptr) {
          result = provide();
          node->right = result;
          node->right->parent = node;
          if (this->header->right == node) this->header->right = node->right;
          maintainAfterInsert(result);
          this->cnt += 1;
        } else {
          result = getNodeOrProvide(node->right, key, provide);
        }
      }

      return result;
    }

    Node* getNode(Node* node, K key) const {
      assert(node != nullptr);

      if (key == node->first) {
        return node;
      }

      if (compare(key, node->first)) {
        /* key < node->first */
        return node->left == nullptr ? nullptr : getNode(node->left, key);
      } else {
        /* key > node->first */
        return node->right == nullptr ? nullptr : getNode(node->right, key);
      }
    }

    void insert(Node*& node, K key, V value, bool replace = true) {
      assert(node != nullptr);

      if (key == node->first) {
        if (replace) {
          node->second = value;
        }
        return;
      }

      if (compare(key, node->first)) {
        /* key < node->first */
        if (node->left == nullptr) {
          node->left = new Node(key, value);
          node->left->parent = node;
          if (this->header->left == node) this->header->left = node->left;
          maintainAfterInsert(node->left);
          this->cnt += 1;
        } else {
          insert(node->left, key, value, replace);
        }
      } else {
        /* key > node->first */
        if (node->right == nullptr) {
          node->right = new Node(key, value);
          node->right->parent = node;
          if (this->header->right == node) this->header->right = node->right;
          maintainAfterInsert(node->right);
          this->cnt += 1;
        } else {
          insert(node->right, key, value, replace);
        }
      }
    }

    void maintainAfterRemove(const Node* node) {
      if (isRoot(node)) {
        return;
      }

      assert(node->isBlack() && hasSibling(node));

      Direction direction = getDirection(node);

      Node* sibling = getSibling(node);
      if (sibling->isRed()) {
        // Case 1: Sibling is RED, parent and nephews must be BLACK
        //   Step 1. If N is a left child, left rotate P;
        //           If N is a right child, right rotate P.
        //   Step 2. Paint S to BLACK, P to RED
        //   Step 3. Goto Case 2, 3, 4, 5
        //      [P]                   <S>               [S]
        //      / \    l-rotate(P)    / \    repaint    / \
        //    [N] <S>  ==========>  [P] [D]  ======>  <P> [D]
        //        / \               / \               / \
        //      [C] [D]           [N] [C]           [N] [C]
        Node* parent = node->parent;
        assert(parent != nullptr && parent->isBlack());
        assert(sibling->left != nullptr && sibling->left->isBlack());
        assert(sibling->right != nullptr && sibling->right->isBlack());
        // Step 1
        rotateSameDirection(node->parent, direction);
        // Step 2
        sibling->color = Node::BLACK;
        parent->color = Node::RED;
        sibling = getSibling(node);
      }

      Node* closeNephew = direction == Direction::LEFT ? sibling->left : sibling->right;
      Node* distantNephew = direction == Direction::LEFT ? sibling->right : sibling->left;

      bool closeNephewIsBlack = closeNephew == nullptr || closeNephew->isBlack();
      bool distantNephewIsBlack = distantNephew == nullptr || distantNephew->isBlack();

      assert(sibling->isBlack());

      if (closeNephewIsBlack && distantNephewIsBlack) {
        if (node->parent->isRed()) {
          // Case 2: Sibling and nephews are BLACK, parent is RED
          //   Swap the color of P and S
          //      <P>             [P]
          //      / \             / \
          //    [N] [S]  ====>  [N] <S>
          //        / \             / \
          //      [C] [D]         [C] [D]
          sibling->color = Node::RED;
          node->parent->color = Node::BLACK;
          return;
        } else {
          // Case 3: Sibling, parent and nephews are all black
          //   Step 1. Paint S to RED
          //   Step 2. Recursively maintain P
          //      [P]             [P]
          //      / \             / \
          //    [N] [S]  ====>  [N] <S>
          //        / \             / \
          //      [C] [D]         [C] [D]
          sibling->color = Node::RED;
          maintainAfterRemove(node->parent);
          return;
        }
      } else {
        if (!closeNephewIsBlack) {
          // Case 4: Sibling is BLACK, close nephew is RED,
          //         distant nephew is BLACK
          //   Step 1. If N is a left child, right rotate P;
          //           If N is a right child, left rotate P.
          //   Step 2. Swap the color of close nephew and sibling
          //   Step 3. Goto case 5
          //                            {P}                {P}
          //      {P}                   / \                / \
          //      / \    r-rotate(S)  [N] <C>   repaint  [N] [C]
          //    [N] [S]  ==========>        \   ======>        \
          //        / \                     [S]                <S>
          //      <C> [D]                     \                  \
          //                                  [D]                [D]
          // Step 1
          rotateOppositeDirection(sibling, direction);
          // Step 2
          closeNephew->color = Node::BLACK;
          sibling->color = Node::RED;
          // Update sibling and nephews after rotation
          sibling = getSibling(node);
          closeNephew = direction == Direction::LEFT ? sibling->left : sibling->right;
          distantNephew = direction == Direction::LEFT ? sibling->right : sibling->left;
        }

        // Case 5: Sibling is BLACK, close nephew is BLACK,
        //         distant nephew is RED
        //      {P}                   [S]
        //      / \    l-rotate(P)    / \
        //    [N] [S]  ==========>  {P} <D>
        //        / \               / \
        //      [C] <D>           [N] [C]
        assert(closeNephew == nullptr || closeNephew->isBlack());
        assert(distantNephew->isRed());
        // Step 1
        rotateSameDirection(node->parent, direction);
        // Step 2
        sibling->color = node->parent->color;
        node->parent->color = Node::BLACK;
        if (distantNephew != nullptr) {
          distantNephew->color = Node::BLACK;
        }
        return;
      }
    }

    template <typename NodeConsumerF> bool remove(Node* node, K key, NodeConsumerF&& action) {
      assert(node != nullptr);

      if (key != node->first) {
        if (compare(key, node->first)) {
          /* key < node->first */
          Node*& left = node->left;
          if (left != nullptr && remove(left, key, action)) {
            maintainRelationship(node);
            return true;
          } else {
            return false;
          }
        } else {
          /* key > node->first */
          Node*& right = node->right;
          if (right != nullptr && remove(right, key, action)) {
            maintainRelationship(node);
            return true;
          } else {
            return false;
          }
        }
      }

      assert(key == node->first);
      if (this->header->left == node) {
        if (node->right != nullptr) {
          Node* l = node->right;
          while (l->left != nullptr) l = l->left;
          this->header->left = l;
        } else {
          this->header->left = node->parent;
        }
      }
      if (this->header->right == node) {
        if (node->left != nullptr) {
          Node* r = node->left;
          while (r->right != nullptr) r = r->right;
          this->header->right = r;
        } else {
          this->header->right = node->parent;
        }
      }
      action(node);

      if (this->size() == 1) {
        // Current node is the only node of the tree
        this->clear();
        return true;
      }

      if (node->left != nullptr && node->right != nullptr) {
        // Case 1: If the node is strictly internal
        //   Step 1. Find the successor S with the smallest key
        //           and its parent P on the right subtree.
        //   Step 2. Swap S and N, S is the node that will be
        //           deleted in place of N.
        //   Step 3. N = S, goto Case 2, 3
        //     |                    |
        //     N                    S
        //    / \                  / \
        //   L  ..   swap(N, S)   L  ..
        //       |   =========>       |
        //       P                    P
        //      / \                  / \
        //     S  ..                N  ..

        // Step 1
        Node* successor = node->right;
        Node* parent = node;
        while (successor->left != nullptr) {
          parent = successor;
          successor = parent->left;
        }
        // Step 2
        switch (getDirection(successor)) {
          case Direction::LEFT:
            successor->parent->left = node;
            break;
          case Direction::RIGHT:
            successor->parent->right = node;
            break;
        }
        switch (getDirection(node)) {
          case Direction::LEFT:
            node->parent->left = successor;
            break;
          case Direction::RIGHT:
            node->parent->right = successor;
            break;
          case Direction::ROOT:
            this->root = this->header->parent = successor;
            break;
        }
        std::swap(node->parent, successor->parent);
        std::swap(node->left, successor->left);
        std::swap(node->right, successor->right);
        std::swap(node->color, successor->color);
        maintainRelationship(node);
        maintainRelationship(successor);
        maintainRelationship(parent);
      }

      if (node->left == nullptr && node->right == nullptr) {
        // Current node must not be the root
        assert(!isRoot(node));

        // Case 2: Current node is a leaf
        //   Step 1. Unlink and remove it.
        //   Step 2. If N is BLACK, maintain N;
        //           If N is RED, do nothing.

        // The maintain operation won't change the node itself,
        //  so we can perform maintain operation before unlink the node.
        if (node->isBlack()) {
          maintainAfterRemove(node);
        }
        if (getDirection(node) == Direction::LEFT) {
          delete node->parent->left;
          node->parent->left = nullptr;
        } else {
          delete node->parent->right;
          node->parent->right = nullptr;
        }
      } else {
        assert(node->left == nullptr || node->right == nullptr);
        // Case 3: Current node has a single left or right child
        //   Step 1. Replace N with its child
        //   Step 2. If N is BLACK, maintain N
        Node* parent = node->parent;
        Node* replacement = (node->left != nullptr ? node->left : node->right);
        bool black = node->isBlack();
        switch (getDirection(node)) {
          case Direction::ROOT:
            this->root = this->header->parent = replacement;
            break;
          case Direction::LEFT:
            parent->left = replacement;
            break;
          case Direction::RIGHT:
            parent->right = replacement;
            break;
        }
        delete replacement->parent;
        replacement->parent = parent;

        if (black) {
          if (replacement->isRed()) {
            replacement->color = Node::BLACK;
          } else {
            maintainAfterRemove(replacement);
          }
        }
      }

      this->cnt -= 1;
      return true;
    }

    template <typename NodeConsumerF> void inorderTraversal(NodeConsumerF&& action) const {
      if (this->root == nullptr) {
        return;
      }

      std::stack<Node*> stack;
      Node* node = this->root;

      while (node != nullptr || !stack.empty()) {
        while (node != nullptr) {
          stack.push(node);
          node = node->left;
        }
        if (!stack.empty()) {
          node = stack.top();
          stack.pop();
          action(node);
          node = node->right;
        }
      }
    }
  };
}  // namespace zs
