import os
import time

import torch
import pygame
import cairosvg
import numpy as np
from treys import Card
from poker_env import Action, ObsProcessor
from poker_env import HeadsUpPoker, AlwaysCallPlayer, RandomPlayer

from train_v0 import SimpleModel

pygame.init()

# Set up the drawing window
WIDTH, HEIGHT = 1280, 960
CARD_WIDTH, CARD_HEIGHT = 100, 150
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 50
FONT_SIZE = 24
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Heads-Up Poker")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)


def card_str_to_rank(card_str):
    r = card_str[0]
    if r == "T":
        return "10"
    if r == "J":
        return "jack"
    if r == "Q":
        return "queen"
    if r == "K":
        return "king"
    if r == "A":
        return "ace"
    return r


def card_str_to_suit(card_str):
    s = card_str[1].upper()
    if s == "C":
        return "clubs"
    if s == "D":
        return "diamonds"
    if s == "H":
        return "hearts"
    if s == "S":
        return "spades"
    raise ValueError(f"Invalid suit: {s}")


def card_str_to_image_path(card_str):
    return f"playing-cards-assets/svg-cards/{card_str_to_rank(card_str)}_of_{card_str_to_suit(card_str)}.svg"


def svg_to_png(svg_path, png_path):
    cairosvg.svg2png(url=svg_path, write_to=png_path)


def get_back_sprite():
    return pygame.image.load("playing-cards-assets/png/back.png")


class PlayerStatistics:
    def __init__(self):
        self.rewards = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def get_avg_reward(self):
        if not self.rewards:
            return 0
        return sum(self.rewards) / len(self.rewards)

    def get_hands_played(self):
        return len(self.rewards)

    def get_last_reward(self):
        if not self.rewards:
            return 0
        return self.rewards[-1]

    def draw(self, screen):
        font = pygame.font.Font(None, FONT_SIZE)
        text = font.render(f"Last reward: {self.get_last_reward()}", True, WHITE)
        screen.blit(text, (0, HEIGHT - 3 * FONT_SIZE))

        text = font.render(f"Hands played: {self.get_hands_played()}", True, WHITE)
        screen.blit(text, (0, HEIGHT - 2 * FONT_SIZE))

        text = font.render(f"Average reward: {self.get_avg_reward(): .2f}", True, WHITE)
        screen.blit(text, (0, HEIGHT - FONT_SIZE))


class Game:
    def __init__(self, env):
        self.env = env
        self.state = None

    def reset(self):
        self.state = self.env.reset()

    def get_player_idx(self):
        return 1 - int(self.env.is_player_dealer)

    def _convert_cards(self, cards):
        return [Card.int_to_str(c) for c in cards]

    def get_player_cards(self):
        return self._convert_cards(self.env.player_hand[self.get_player_idx()])

    def get_opponent_cards(self):
        return self._convert_cards(self.env.player_hand[1 - self.get_player_idx()])

    def opponent_folded(self):
        return (1 - self.get_player_idx()) not in self.env.active_players

    def get_community_cards(self):
        return self._convert_cards(self.env._board())

    def get_pot(self):
        return self.env.pot_size

    def get_stage(self):
        return self.env.stage.name

    def get_player_info(self):
        return (
            self.env.stack_sizes[self.get_player_idx()],
            self.env.bets_this_stage[self.get_player_idx()],
        )

    def get_opponent_info(self):
        return (
            self.env.stack_sizes[1 - self.get_player_idx()],
            self.env.bets_this_stage[1 - self.get_player_idx()],
        )

    def step(self, action):
        self.state, reward, done, _ = self.env.step(action)
        return reward, done


def draw_card(screen, x, y, card=None):
    if card is None:
        card_sprite = get_back_sprite()
    else:
        svg_path = card_str_to_image_path(card)
        png_path = svg_path.replace(".svg", ".png")
        if not os.path.exists(png_path):
            svg_to_png(svg_path, png_path)
        card_sprite = pygame.image.load(png_path)
    card_sprite = pygame.transform.scale(card_sprite, (100, 150))
    screen.blit(card_sprite, (x, y))


def draw_opponent_cards(screen, cards, show_opponent_cards):
    if show_opponent_cards:
        draw_card(screen, WIDTH // 2 - CARD_WIDTH, 0, cards[0])
        draw_card(screen, WIDTH // 2, 0, cards[1])
    else:
        draw_card(screen, WIDTH // 2 - CARD_WIDTH, 0)
        draw_card(screen, WIDTH // 2, 0)


def draw_opponent_distribution(screen, distribution, show_opponent_distribution):
    if show_opponent_distribution and distribution is not None:
        font = pygame.font.Font(None, FONT_SIZE)
        action_short_name = ["FOLD  ", "CALL   ", "RAISE ", "ALL IN"]
        for i, prob in enumerate(distribution):
            text = font.render(f"{action_short_name[i]}: {prob:.2f}", True, WHITE)
            screen.blit(
                text,
                (
                    WIDTH // 2 + CARD_WIDTH,
                    i * FONT_SIZE + CARD_HEIGHT // 2 - 2 * FONT_SIZE,
                ),
            )


def draw_player_cards(screen, cards):
    draw_card(screen, WIDTH // 2 - CARD_WIDTH, HEIGHT - CARD_HEIGHT, cards[0])
    draw_card(screen, WIDTH // 2, HEIGHT - CARD_HEIGHT, cards[1])


def draw_community_cards(screen, cards):
    for i, card in enumerate(cards):
        draw_card(
            screen,
            WIDTH // 2 - CARD_WIDTH * 5 // 2 + i * CARD_WIDTH,
            HEIGHT // 2 - CARD_HEIGHT // 2,
            card,
        )


def draw_opponent_info(screen, stack, bet):
    font = pygame.font.Font(None, FONT_SIZE)
    text = font.render(f"Opponent: {stack} ({bet})", True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2, CARD_HEIGHT + FONT_SIZE))
    screen.blit(text, text_rect)


def draw_player_info(screen, stack, bet):
    font = pygame.font.Font(None, FONT_SIZE)
    text = font.render(f"Player: {stack} ({bet})", True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - CARD_HEIGHT - FONT_SIZE))
    screen.blit(text, text_rect)


def draw_pot_and_stage(screen, pot, stage):
    font = pygame.font.Font(None, FONT_SIZE)
    text = font.render(f"Pot: {pot}", True, WHITE)
    text_rect = text.get_rect(
        center=(WIDTH // 2, HEIGHT // 2 - CARD_HEIGHT // 2 - 2 * FONT_SIZE)
    )
    screen.blit(text, text_rect)

    text = font.render(f"Stage: {stage}", True, WHITE)
    text_rect = text.get_rect(
        center=(WIDTH // 2, HEIGHT // 2 - CARD_HEIGHT // 2 - FONT_SIZE)
    )
    screen.blit(text, text_rect)


class Button:
    def __init__(self, x, y, width, height, text, color, action):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.action = action

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        font = pygame.font.Font(None, FONT_SIZE)
        text = font.render(self.text, True, BLACK)
        text_rect = text.get_rect(
            center=(self.x + self.width // 2, self.y + self.height // 2)
        )
        screen.blit(text, text_rect)

    def is_clicked(self, x, y):
        return (
            self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height
        )


def draw_buttons(screen, buttons):
    for button in buttons:
        button.draw(screen)


def show_opponent_folded():
    font = pygame.font.Font(None, FONT_SIZE)
    text = font.render("Opponent folded", True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2, CARD_HEIGHT + FONT_SIZE * 2))
    screen.blit(text, text_rect)


def draw_table(screen):
    BORDER_SIZE = 20
    TABLE_WIDTH, TABLE_HEIGHT = 800, 400
    pygame.draw.ellipse(
        screen,
        BLACK,
        (
            WIDTH // 2 - TABLE_WIDTH // 2 - BORDER_SIZE // 2,
            HEIGHT // 2 - TABLE_HEIGHT // 2 - BORDER_SIZE // 2,
            TABLE_WIDTH + BORDER_SIZE,
            TABLE_HEIGHT + BORDER_SIZE,
        ),
    )
    pygame.draw.ellipse(
        screen,
        GREEN,
        (
            WIDTH // 2 - TABLE_WIDTH // 2,
            HEIGHT // 2 - TABLE_HEIGHT // 2,
            TABLE_WIDTH,
            TABLE_HEIGHT,
        ),
    )


class CFRPlayer:
    def __init__(self):
        from deepcfr.model import BaseModel
        from deepcfr.player_wrapper import PolicyPlayerWrapper

        model = BaseModel().cuda()
        model.load_state_dict(torch.load("deepcfr/policy.pth", weights_only=True))
        self.player = PolicyPlayerWrapper(model)

    @property
    def previous_action_distribution(self):
        return self.player.previous_action_distribution

    def __call__(self, obs):
        return self.player(obs)


def main():
    obs_processor = ObsProcessor()
    player = CFRPlayer()
    env = HeadsUpPoker(obs_processor, player)
    game = Game(env)
    game.reset()

    running = True
    show_opponent_cards = False
    show_opponent_distribution = False
    buttons = [
        Button(
            WIDTH - 4 * (BUTTON_WIDTH + 10),
            HEIGHT - BUTTON_HEIGHT - 50,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "Fold",
            RED,
            Action.FOLD,
        ),
        Button(
            WIDTH - 3 * (BUTTON_WIDTH + 10),
            HEIGHT - BUTTON_HEIGHT - 50,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "Call",
            RED,
            Action.CHECK_CALL,
        ),
        Button(
            WIDTH - 2 * (BUTTON_WIDTH + 10),
            HEIGHT - BUTTON_HEIGHT - 50,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "Raise",
            RED,
            Action.RAISE,
        ),
        Button(
            WIDTH - (BUTTON_WIDTH + 10),
            HEIGHT - BUTTON_HEIGHT - 50,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "All in",
            RED,
            Action.ALL_IN,
        ),
    ]
    show_opponent_cards_button = Button(
        0,
        HEIGHT // 2 - BUTTON_HEIGHT // 2,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
        "Show cards",
        RED,
        None,
    )

    show_opponent_distribution_button = Button(
        0,
        HEIGHT // 2 + BUTTON_HEIGHT,
        BUTTON_WIDTH * 2,
        BUTTON_HEIGHT,
        "Show distribution",
        RED,
        None,
    )

    player_stats = PlayerStatistics()
    while running:
        screen.fill(BLUE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if show_opponent_cards_button.is_clicked(x, y):
                    show_opponent_cards = not show_opponent_cards

                if show_opponent_distribution_button.is_clicked(x, y):
                    show_opponent_distribution = not show_opponent_distribution

                for button in buttons:
                    if button.is_clicked(x, y):
                        reward, done = game.step(button.action)
                        if done:
                            player_stats.add_reward(reward)
                            game.reset()
                        break

        draw_table(screen)
        draw_opponent_cards(screen, game.get_opponent_cards(), show_opponent_cards)
        draw_opponent_distribution(
            screen,
            env.env_player.previous_action_distribution,
            show_opponent_distribution,
        )
        draw_player_cards(screen, game.get_player_cards())
        draw_community_cards(screen, game.get_community_cards())
        draw_buttons(screen, buttons)
        show_opponent_cards_button.draw(screen)
        show_opponent_distribution_button.draw(screen)
        if game.opponent_folded():
            show_opponent_folded()

        player_stack, player_bet = game.get_player_info()
        opponent_stack, opponent_bet = game.get_opponent_info()
        pot, stage = game.get_pot(), game.get_stage()
        draw_opponent_info(screen, opponent_stack, opponent_bet)
        draw_player_info(screen, player_stack, player_bet)
        draw_pot_and_stage(screen, pot, stage)
        player_stats.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
